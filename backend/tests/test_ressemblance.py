"""Ressemblance vocale : échantillon de clonage sans intrus, phrases au timbre dérivé refaites.

Le vrai modèle d'empreinte (ECAPA) est remplacé par une empreinte simple : l'énergie par bande de
fréquence. Une « voix » est ici un son de hauteur donnée ; un intrus a une autre hauteur."""

from contextlib import contextmanager

import numpy as np
import soundfile as sf

from pipeline import ressemblance
from pipeline.voice import Manifeste, choisir_extraits, fichier_brut

TAUX = ressemblance.TAUX
BANDES = [(100, 300), (300, 700), (700, 1500), (1500, 4000)]


def son(frequence, duree, taux=TAUX):
    t = np.arange(int(duree * taux)) / taux
    return (0.2 * np.sin(2 * np.pi * frequence * t)).astype(np.float32)


def empreinte_simple(signal):
    if signal.size < ressemblance.DUREE_MIN_S * TAUX:
        return None
    spectre = np.abs(np.fft.rfft(signal)) ** 2
    f = np.fft.rfftfreq(signal.size, 1 / TAUX)
    v = np.array([spectre[(f >= a) & (f < b)].sum() for a, b in BANDES]) + 1e-3 * spectre.sum()
    v = np.sqrt(v)
    return v / np.linalg.norm(v)


@contextmanager
def faux_modele():
    yield empreinte_simple


def test_scores_ecartent_l_intrus(tmp_path):
    # Trois phrases du locuteur (220 Hz) et une phrase attribuée à tort (2 000 Hz)
    source = tmp_path / "voix.wav"
    sf.write(source, np.concatenate([son(220, 3), son(230, 3), son(2000, 3), son(225, 3)]), TAUX)
    segments = [{"id": i, "debut": 3.0 * i, "fin": 3.0 * i + 3} for i in range(4)]
    scores = ressemblance.scores_candidats(source, segments, empreinte_simple)
    assert set(scores) == {0, 1, 2, 3}
    assert min(scores, key=scores.get) == 2
    assert scores[2] < ressemblance.seuil_echantillon(scores) <= min(scores[0], scores[1], scores[3])


def test_echantillon_sans_intrus_et_meilleur_en_tete():
    segments = [{"id": 0, "debut": 0.0, "fin": 5.0}, {"id": 1, "debut": 10.0, "fin": 15.0},
                {"id": 2, "debut": 20.0, "fin": 31.0}, {"id": 3, "debut": 40.0, "fin": 44.0}]
    scores = {0: 0.80, 1: 0.95, 2: 0.30, 3: 0.85}  # la plus longue (id 2) n'est pas sa voix
    choisis = choisir_extraits(segments, scores=scores)
    assert [s["id"] for s in choisis] == [1, 3, 0]


def test_echantillon_garde_l_intrus_faute_de_mieux():
    segments = [{"id": 0, "debut": 0.0, "fin": 3.0}, {"id": 1, "debut": 10.0, "fin": 20.0}]
    choisis = choisir_extraits(segments, scores={0: 0.9, 1: 0.2})
    assert {s["id"] for s in choisis} == {0, 1}  # sans l'intrus, moins de 6 s : il reste


def test_phrases_a_refaire():
    assert ressemblance.phrases_a_refaire({1: 0.7, 2: 0.72, 3: 0.4}) == set()  # 3 phrases : 30 % → aucune
    scores = {i: 0.7 for i in range(10)} | {10: 0.4, 11: 0.65}
    assert ressemblance.phrases_a_refaire(scores) == {10}  # 0,65 : un peu sous la médiane, pas dérivée
    scores = {i: 0.7 for i in range(4)} | {i: 0.2 for i in range(4, 10)}
    assert len(ressemblance.phrases_a_refaire(scores)) <= 3  # jamais plus de 30 % des phrases
    assert ressemblance.phrases_a_refaire({1: 0.9, 2: 0.9, 3: 0.9, 4: 0.75}) == set()  # 0,75 : ressemblante


def test_verifier_ressemblance_refait_la_phrase_derivee(tmp_path, monkeypatch):
    import pipeline.voice as voice
    from pipeline import runner

    dossier = tmp_path / "voix"
    dossier.mkdir()
    reference = dossier / "reference_v3_A.wav"
    sf.write(reference, son(220, 4), TAUX)
    segments = [{"id": i, "locuteur": "A", "traduction": f"phrase {i}", "debut": 2.0 * i, "fin": 2.0 * i + 2}
                for i in range(5)]
    for seg in segments:
        sf.write(fichier_brut(dossier, seg["id"]), son(2000 if seg["id"] == 3 else 225, 2), TAUX)
    appels = []

    def fausse_generation(segs, langue, voix, dossier_, on_progress=None, moteur=None):
        appels.append([(s["id"], s.get("stabiliser")) for s in segs])
        for s in segs:
            sf.write(fichier_brut(dossier_, s["id"]), son(222, 2), TAUX)  # cette fois, la bonne voix
        return len(segs)

    monkeypatch.setattr(ressemblance, "modele_empreinte", faux_modele)
    monkeypatch.setattr(voice, "generer_voix", fausse_generation)
    resultat = runner._verifier_ressemblance("job", segments, {"A": f"ref:{reference}"},
                                             {s["id"]: f"ref:{reference}" for s in segments}, "fr", dossier,
                                             lambda f: None)
    assert appels == [[(3, True)]]  # seule la phrase dérivée, refaite en mode stable
    assert resultat[3] > 0.9 and segments[3]["stabiliser"] is True
    assert not list(dossier.glob("avant_*.wav"))


def test_verifier_ressemblance_garde_l_ancienne_si_la_reprise_est_pire(tmp_path, monkeypatch):
    import pipeline.voice as voice
    from pipeline import runner

    dossier = tmp_path / "voix"
    dossier.mkdir()
    reference = dossier / "reference_v3_A.wav"
    sf.write(reference, son(220, 4), TAUX)
    segments = [{"id": i, "locuteur": "A", "traduction": "x", "debut": 0.0, "fin": 2.0} for i in range(5)]
    for seg in segments:
        derive = son(225, 2) + 1.75 * son(1000, 2)  # timbre à moitié parti ailleurs
        sf.write(fichier_brut(dossier, seg["id"]), derive if seg["id"] == 3 else son(225, 2), TAUX)
    avant = sf.read(fichier_brut(dossier, 3))[0]

    def pire(segs, langue, voix, dossier_, on_progress=None, moteur=None):
        for s in segs:
            sf.write(fichier_brut(dossier_, s["id"]), son(3000, 2), TAUX)
        return len(segs)

    monkeypatch.setattr(ressemblance, "modele_empreinte", faux_modele)
    monkeypatch.setattr(voice, "generer_voix", pire)
    runner._verifier_ressemblance("job", segments, {"A": f"ref:{reference}"}, {i: f"ref:{reference}" for i in range(5)},
                                  "fr", dossier, lambda f: None)
    assert np.allclose(sf.read(fichier_brut(dossier, 3))[0], avant)  # la version d'origine est remise


def test_voix_transformee_pas_verifiee(tmp_path, monkeypatch):
    from pipeline import runner

    monkeypatch.setattr(ressemblance, "modele_empreinte", faux_modele)
    # Voix d'enfant volontaire (hauteur changée) ou voix de la bibliothèque : pas de comparaison à l'origine
    assert runner._verifier_ressemblance("job", [], {"A": "ref:x.wav|pitch=1.4", "B": "xtts:Daisy"}, {}, "fr",
                                         tmp_path, lambda f: None) == {}


def test_phrase_stabilisee_moins_libre_et_sans_intonation(tmp_path):
    from pipeline.voice import intonation, temperature

    seg = {"id": 1, "ton": "crié", "intonation": "x.wav", "traduction": "a", "debut": 0, "fin": 1}
    assert intonation(seg) == "x.wav"
    seg["stabiliser"] = True
    assert intonation(seg) is None and temperature(seg) <= 0.55
    # Une phrase non stabilisée garde sa clé de cache d'avant (pas de régénération inutile)
    manifeste = Manifeste(tmp_path)
    assert manifeste._cle({**seg, "stabiliser": False}, "fr", "v")["stable"] is None
