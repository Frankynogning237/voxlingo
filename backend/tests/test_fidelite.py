"""Fidélité à la voix d'origine : volume au décibel près, timbre, échantillon de clonage, traduction."""

import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from pipeline import niveau, timbre_eq, translate
from pipeline.voice import choisir_extraits

TAUX = 24000
ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg absent")


def sinus(amplitude, duree=1.0, frequence=220.0, taux=TAUX):
    t = np.arange(int(duree * taux)) / taux
    return (amplitude * np.sin(2 * np.pi * frequence * t)).astype(np.float32)


# ── Volume ────────────────────────────────────────────────────────────────

def test_niveau_actif_ignore_les_silences():
    voix = sinus(0.1)
    avec_silences = np.concatenate([np.zeros(TAUX), voix, np.zeros(2 * TAUX)])
    attendu = 20 * np.log10(0.1 / np.sqrt(2))  # ≈ -23,0 dBFS
    assert niveau.niveau_actif_db(voix, TAUX) == pytest.approx(attendu, abs=0.2)
    assert niveau.niveau_actif_db(avec_silences, TAUX) == pytest.approx(attendu, abs=0.2)
    assert niveau.niveau_actif_db(np.zeros(TAUX, dtype=np.float32), TAUX) is None


def test_gain_pour_egaler():
    assert niveau.gain_pour_egaler(-20.0, -26.5) == 6.5
    assert niveau.gain_pour_egaler(-20.0, -80.0) == niveau.GAIN_MAX_DB  # mesure suspecte : plafonnée
    assert niveau.gain_pour_egaler(None, -20.0) is None


def test_lire_extrait_sans_tout_charger(tmp_path):
    fichier = tmp_path / "voix.wav"
    sf.write(fichier, np.concatenate([sinus(0.05, 2.0), sinus(0.4, 1.0)]), TAUX)
    signal, taux = niveau.lire_extrait(fichier, 2.0, 3.0)
    assert taux == TAUX and len(signal) == TAUX
    assert niveau.niveau_actif_db(signal, taux) == pytest.approx(20 * np.log10(0.4 / np.sqrt(2)), abs=0.2)


@ffmpeg
def test_caler_amene_la_voix_au_niveau_d_origine(tmp_path):
    from pipeline.align import caler

    brut, cale = tmp_path / "brut.wav", tmp_path / "cale.wav"
    sf.write(brut, np.concatenate([sinus(0.02, 1.5), np.zeros(TAUX // 2, dtype=np.float32)]), TAUX)
    seg = {"id": 1, "debut": 0.0, "fin": 2.0}
    resultat = caler(brut, cale, seg, gain_db=0.0, egaliseur=None, niveau_cible_db=-18.0)
    assert niveau.niveau_fichier_db(cale) == pytest.approx(-18.0, abs=0.3)  # au décibel près
    assert resultat["gain_db"] > 10
    assert not list(tmp_path.glob("*_tmp.wav"))


# ── Timbre ────────────────────────────────────────────────────────────────

def bruit_colore(pente_db_par_octave, duree=3.0, taux=TAUX, graine=0):
    """Bruit dont le spectre décroît (ou croît) d'une pente donnée : une voix « sourde » ou « brillante »."""
    rng = np.random.default_rng(graine)
    spectre = np.fft.rfft(rng.standard_normal(int(duree * taux)))
    f = np.maximum(np.fft.rfftfreq(int(duree * taux), 1 / taux), 20)
    spectre *= 10 ** (pente_db_par_octave * np.log2(f / 1000) / 20)
    signal = np.fft.irfft(spectre)
    return (0.1 * signal / np.std(signal)).astype(np.float32)


def test_correction_de_timbre_rapproche_les_spectres():
    origine = timbre_eq.spectre_bandes(bruit_colore(+2.0), TAUX)  # voix d'origine plus brillante
    generee = timbre_eq.spectre_bandes(bruit_colore(-2.0, graine=1), TAUX)  # voix générée plus sourde
    gains = timbre_eq.correction(origine, generee)
    assert len(gains) == len(timbre_eq.BANDES_HZ)
    assert gains[-1] > gains[0] + 4  # on remonte les aigus, on baisse les graves
    assert abs(float(np.mean(gains))) < 1.5  # correction centrée : le volume ne bouge pas
    assert float(np.max(np.abs(gains))) <= timbre_eq.CORRECTION_MAX_DB
    filtre = timbre_eq.filtre_ffmpeg(gains)
    assert filtre.startswith("firequalizer=gain_entry='entry(80,") and filtre.count("entry(") == len(gains)


def test_pas_de_correction_quand_les_timbres_sont_proches():
    spectre = timbre_eq.spectre_bandes(bruit_colore(0.0), TAUX)
    assert timbre_eq.filtre_ffmpeg(timbre_eq.correction(spectre, spectre)) is None
    assert timbre_eq.filtre_ffmpeg(None) is None


@ffmpeg
def test_filtre_de_timbre_accepte_par_ffmpeg(tmp_path):
    import subprocess

    entree, sortie = tmp_path / "e.wav", tmp_path / "s.wav"
    sf.write(entree, bruit_colore(-2.0), TAUX)
    gains = timbre_eq.correction(timbre_eq.spectre_bandes(bruit_colore(2.0), TAUX),
                                 timbre_eq.spectre_bandes(bruit_colore(-2.0, graine=1), TAUX))
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(entree), "-af", timbre_eq.filtre_ffmpeg(gains),
                        str(sortie)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    apres = timbre_eq.spectre_bandes(sf.read(sortie, dtype="float32")[0], TAUX)
    avant = timbre_eq.spectre_bandes(bruit_colore(-2.0), TAUX)
    assert (apres[-1] - apres[0]) > (avant[-1] - avant[0]) + 4  # la voix est devenue plus brillante


# ── Échantillon de clonage ────────────────────────────────────────────────

def test_echantillon_evite_les_chevauchements():
    moi = [{"debut": 0.0, "fin": 9.0}, {"debut": 20.0, "fin": 25.0}, {"debut": 30.0, "fin": 34.0}]
    autre = [{"debut": 8.0, "fin": 10.0}]  # quelqu'un parle par-dessus la phrase la plus longue
    choisis = choisir_extraits(moi, autre)
    assert {"debut": 0.0, "fin": 9.0} not in choisis
    assert choisis == [{"debut": 20.0, "fin": 25.0}, {"debut": 30.0, "fin": 34.0}]
    # Sans autre choix, la phrase qui chevauche reste utilisable
    assert choisir_extraits([{"debut": 0.0, "fin": 9.0}], autre) == [{"debut": 0.0, "fin": 9.0}]


# ── Traduction ────────────────────────────────────────────────────────────

def test_genre_selon_voix():
    assert translate.genre_selon_voix(120) == "a man"
    assert translate.genre_selon_voix(220) == "a woman"
    assert translate.genre_selon_voix(170) is None  # zone ambiguë : on ne devine pas
    assert translate.genre_selon_voix(320) is None  # enfant : genre incertain
    assert translate.genre_selon_voix(None) is None


def test_genre_transmis_au_traducteur():
    seg = {"id": 0, "texte": "Je suis prête", "locuteur": "SPEAKER_01", "debut": 0, "fin": 1}
    with translate.contexte_de_la_scene(noms={"SPEAKER_01": "Claire"}, genres={"SPEAKER_01": "a woman"}):
        lignes = translate._lignes_de_scene(seg)
    assert lignes[0] == "Speaker: Claire (a woman)"
    assert "grammatical gender" in lignes[1]


class FauxClient:
    def __init__(self, reponse):
        self.reponse, self.prompts = reponse, []

    def post(self, chemin, json):
        self.prompts.append(json["messages"][1]["content"])
        contenu = self.reponse

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": contenu}}

        return R()


SEGMENTS = [{"texte": "Bonjour Paul."}, {"texte": "Tu viens ce soir ?"}, {"texte": "Oui, je viens."}]


def test_relecture_corrige_une_ligne():
    client = FauxClient(json.dumps({"translation": "Kommst du heute Abend?"}))
    traductions = ["Hallo Paul.", "Kommen Sie heute Abend?", "Ja, ich komme."]
    assert translate.relire(client, "qwen3:14b", SEGMENTS, traductions, 1, "fr", "de") == "Kommst du heute Abend?"
    assert ">>> \"Tu viens ce soir ?\"" in client.prompts[0] and "Hallo Paul." in client.prompts[0]


def test_relecture_refuse_une_correction_trop_longue_ou_vide():
    traductions = ["Hallo Paul.", "Kommst du heute Abend?", "Ja, ich komme."]
    longue = FauxClient(json.dumps({"translation": "Kommst du heute Abend zu der großen Feier bei meinen Eltern?"}))
    assert translate.relire(longue, "m", SEGMENTS, traductions, 1, "fr", "de") == traductions[1]
    vide = FauxClient(json.dumps({"translation": ""}))
    assert translate.relire(vide, "m", SEGMENTS, traductions, 1, "fr", "de") == traductions[1]
    cassee = FauxClient("pas du JSON")
    assert translate.relire(cassee, "m", SEGMENTS, traductions, 1, "fr", "de") == traductions[1]
