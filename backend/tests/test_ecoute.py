"""Reconnaissance vocale et clonage méticuleux : hallucinations de Whisper écartées, échantillon de clonage
découpé au minutage des mots, écoute de contrôle des voix générées (reprise des phrases infidèles)."""

from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from pipeline import ecoute, runner
from pipeline.ecoute import fidelite, phrases_a_refaire
from pipeline.transcribe import DEPOTS, est_hallucination, termes_du_glossaire
from pipeline.voice import passages_parles


def test_fidelite():
    texte = "Na los, du hast dich total verplappert."
    assert fidelite(texte, "na los du hast dich total verplappert") == 1.0  # ponctuation, casse
    assert fidelite("Ça va très bien", "ca va tres bien") == 1.0  # accents
    assert fidelite("Ich habe 3 Brüder", "Ich habe drei Brüder") > 0.8  # chiffre écrit en lettres : pas refaite
    assert fidelite(texte, "Na, du bist total verploppert!") < 0.65  # mots sautés et déformés
    assert fidelite(texte, texte + " bla bla bla bla bla bla") < 0.8  # syllabes inventées en fin de phrase


def test_reprises_bornees():
    assert phrases_a_refaire({1: 0.9, 2: 0.4, 3: 0.95}) == {2}
    toutes_mauvaises = {i: 0.3 for i in range(10)}
    assert len(phrases_a_refaire(toutes_mauvaises)) == 4  # au plus 40 %
    assert len(phrases_a_refaire({1: 0.3, 2: 0.2, 3: 0.9})) == 2  # petit projet : au moins 2 reprises


@pytest.mark.parametrize("texte,ns,lp,cr,attendu", [
    ("Sous-titres réalisés par la communauté d'Amara.org", 0.1, -0.3, 1.2, True),
    ("Merci d'avoir regardé !", 0.2, -0.4, 1.1, True),
    ("Untertitel im Auftrag des ZDF", 0.1, -0.2, 1.0, True),
    ("Euh…", 0.8, -1.4, 0.8, True),  # silence transcrit sans assurance
    ("oui " * 40, 0.1, -0.3, 3.5, True),  # boucle de répétition
    ("Tu n'entends pas, donc c'est terminé.", 0.05, -0.25, 1.3, False),
    ("Merci, à demain.", 0.1, -0.3, 1.0, False),
])
def test_hallucinations(texte, ns, lp, cr, attendu):
    assert est_hallucination(texte, ns, lp, cr) is attendu


def test_repetition_de_la_phrase_precedente():
    phrase = "Je t'invite à dîner avec moi ce soir."
    assert est_hallucination(phrase, 0.1, -0.3, 1.2, precedent=phrase)
    assert not est_hallucination("Oui.", 0.1, -0.3, 1.0, precedent="Oui.")  # réponse courte répétée : normale


def test_turbo_par_ses_depots_valides():
    assert DEPOTS["large-v3-turbo"][0] == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert termes_du_glossaire("Niska = Niska\nLéon → Leon\nsans séparateur") == ["Niska", "Léon"]


def test_passages_parles_sans_silences():
    mots = [{"mot": "Il", "debut": 1.0, "fin": 1.2}, {"mot": "ya", "debut": 4.0, "fin": 4.3},
            {"mot": "quoi", "debut": 4.4, "fin": 4.7}]
    seg = {"debut": 0.9, "fin": 4.8, "mots": mots}
    passages = passages_parles(seg)
    assert len(passages) == 2  # la pause de 2,8 s est retirée de l'échantillon
    assert passages[0][0] == pytest.approx(0.94) and passages[1][1] == pytest.approx(4.82)
    assert passages_parles({"debut": 2.0, "fin": 3.0}) == [(2.0, 3.0)]  # sans minutage : phrase entière


def test_ecoute_refait_les_phrases_infideles_et_garde_la_meilleure(tmp_path, monkeypatch):
    """Phrase 1 bafouillée : refaite, la reprise (fidèle) est gardée. Phrase 2 : reprise pire, l'originale
    est remise. Phrase 0 fidèle : jamais refaite."""
    textes = {0: "Was muss ich tun, um dich zu retten?", 1: "Na los, du hast dich total verplappert.",
              2: "Hör auf ihn, großer Bruder, bitte jetzt."}
    segments = [{"id": i, "traduction": t, "debut": 0.0, "fin": 2.0} for i, t in textes.items()]
    for i in textes:
        sf.write(tmp_path / f"brut_{i:05d}.wav", np.zeros(2400, np.float32) + i * 0.001, 24000)
    entendus = {0: [textes[0]], 1: ["Na, du bist total verploppert!", textes[1]],
                2: ["Bruder.", "Hör bla bla."]}

    @contextmanager
    def faux_modele():
        yield lambda fichier, langue: entendus[int(fichier.stem[-5:])].pop(0)

    reprises = []

    def faux_generer(segs, langue, voix, dossier, on_progress=None):
        for s in segs:
            reprises.append((s["id"], s.get("stabiliser")))
            sf.write(dossier / f"brut_{s['id']:05d}.wav", np.ones(2400, np.float32) * 0.5, 24000)
        return len(segs)

    monkeypatch.setattr(ecoute, "modele_ecoute", faux_modele)
    monkeypatch.setattr("pipeline.voice.generer_voix", faux_generer)
    fidelites = runner._verifier_ecoute("essai", segments, {}, "de", tmp_path, {0, 1, 2}, lambda f: None)

    assert sorted(i for i, _ in reprises) == [1, 2] and all(stable for _, stable in reprises)
    assert fidelites[0] == 1.0 and fidelites[1] == 1.0
    assert fidelites[2] < 0.65
    assert sf.read(tmp_path / "brut_00001.wav")[0].max() == pytest.approx(0.5, abs=1e-3)  # reprise gardée
    assert sf.read(tmp_path / "brut_00002.wav")[0].max() < 0.01  # originale remise (plus fidèle)
    assert not list(tmp_path.glob("avant_ecoute_*"))


def test_reglages_audio_rétablis_si_preprocessor_manquant():
    """Téléchargement interrompu (preprocessor_config.json absent) : 80 bandes au lieu de 128 → recalées."""
    from types import SimpleNamespace

    from faster_whisper.feature_extractor import FeatureExtractor

    from pipeline.transcribe import _reglages_audio_coherents

    modele = SimpleNamespace(model=SimpleNamespace(n_mels=128), feature_extractor=FeatureExtractor(feature_size=80),
                             feat_kwargs={"feature_size": 80})
    assert _reglages_audio_coherents(modele).feature_extractor.mel_filters.shape[0] == 128
