"""Tests des réglages de timbre et de la bibliothèque de voix."""

import shutil

import numpy as np
import soundfile as sf

from pipeline.voice_bank import VOIX_BIBLIOTHEQUE, filtres_timbre, nettoyer_timbre, separer_spec, transformer


def test_nettoyer_timbre_borne_et_ignore_le_neutre():
    assert nettoyer_timbre({"ton": 30, "formant": 1.0, "graves": 0, "aigus": -4, "inconnu": 2}) == {"ton": 12.0, "aigus": -4.0}
    assert nettoyer_timbre(None) == {}


def test_filtres_timbre():
    assert filtres_timbre({}) == []
    filtres = filtres_timbre({"formant": 1.1, "ton": 2, "telephone": 1})
    assert filtres[0].startswith("asetrate=26400")  # résonances relevées de 10 %
    assert "formant=preserved" in filtres[1]
    assert any("lowpass=f=3400" in f for f in filtres)
    assert filtres[-1].startswith("alimiter")


def test_transformer_change_la_hauteur_sans_changer_la_duree(tmp_path):
    import librosa

    taux = 24000
    t = np.arange(taux * 2) / taux
    son = 0.3 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 8))
    fichier = tmp_path / "son.wav"
    sf.write(fichier, son, taux)
    copie = tmp_path / "copie.wav"
    shutil.copy(fichier, copie)
    transformer(copie, {"ton": 12.0})
    y, _ = librosa.load(copie, sr=taux)
    f0, voise, _ = librosa.pyin(y, fmin=60, fmax=600, sr=taux, frame_length=2048)
    assert abs(np.nanmedian(f0[voise]) - 300) < 10
    assert abs(len(y) / taux - 2.0) < 0.05


def test_bibliotheque_variee():
    assert len(VOIX_BIBLIOTHEQUE) >= 50
    assert len({v["id"] for v in VOIX_BIBLIOTHEQUE}) == len(VOIX_BIBLIOTHEQUE)
    for age in ("enfant", "jeune", "adulte", "senior"):
        for sexe in ("femme", "homme"):
            assert any(v["age"] == age and v["sexe"] == sexe for v in VOIX_BIBLIOTHEQUE)
    assert separer_spec("xtts:Damien Black|ton=-2|formant=0.93") == ("xtts:Damien Black", {"ton": -2.0, "formant": 0.93})
