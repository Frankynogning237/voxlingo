"""Tests du processus Chatterbox (fonctions pures, sans le modèle)."""

import numpy as np
import pytest

from pipeline import chatterbox_worker as cb


def test_intensite_suit_le_ton():
    assert cb.intensite(0.55) < cb.intensite(0.7) < cb.intensite(0.85)
    assert cb.intensite(0.1) == 0.3 and cb.intensite(2.0) == 0.9


def test_echantillon_ignore_la_transformation_d_age():
    assert cb.echantillon("ref:/data/voix/franky.wav|pitch=1.4") == "/data/voix/franky.wav"
    with pytest.raises(ValueError):
        cb.echantillon("xtts:Daisy Studious")


def test_langues():
    assert cb.langue_chatterbox("zh-cn") == "zh" and cb.langue_chatterbox("fr") == "fr"


def test_rogner_silences_garde_une_marge():
    wav = np.zeros(cb.TAUX * 2, dtype=np.float32)
    wav[cb.TAUX // 2: cb.TAUX] = 0.5
    rogne = cb.rogner_silences(wav)
    assert abs(len(rogne) - (cb.TAUX // 2 + 2 * int(0.05 * cb.TAUX))) <= 1
    assert len(cb.rogner_silences(np.zeros(100, dtype=np.float32))) == 100


def _kaggle(tmp_path, monkeypatch, installe: bool):
    """Chatterbox comme sur Kaggle : son dossier existe dès le début de l'installation, « .pret » à la fin."""
    from types import SimpleNamespace

    from pipeline import voice

    racine = tmp_path / "chatterbox"
    (racine / "bin").mkdir(parents=True)
    if installe:
        (racine / ".pret").touch()
    monkeypatch.setattr(voice, "get_settings", lambda: SimpleNamespace(chatterbox_python=str(racine / "bin" / "python")))
    monkeypatch.setattr("config.lire_parametres", lambda: {"moteur_voix": "auto"})
    monkeypatch.setattr(voice, "_chatterbox_pret", False)
    monkeypatch.setattr(voice, "chatterbox_disponible", lambda: (racine / ".pret").is_file())
    monkeypatch.setattr(voice, "SONDAGE_CHATTERBOX_S", 0.01)
    return voice, racine


def test_generation_attend_chatterbox_en_installation(tmp_path, monkeypatch):
    """Sur Kaggle, la génération démarrée avant la fin de l'installation de Chatterbox l'attend au lieu
    de passer à XTTS (voix nettement moins fidèle)."""
    voice, racine = _kaggle(tmp_path, monkeypatch, installe=False)
    attentes = []

    def patienter():
        attentes.append(1)
        if len(attentes) == 3:
            (racine / ".pret").touch()  # installation terminée

    assert voice.moteur_voix("de", patienter) == "chatterbox"
    assert len(attentes) == 3


def test_installation_ratee_passe_a_xtts(tmp_path, monkeypatch):
    voice, _ = _kaggle(tmp_path, monkeypatch, installe=False)
    monkeypatch.setattr(voice, "ATTENTE_CHATTERBOX_S", 0.05)
    assert voice.moteur_voix("de", lambda: None) == "xtts"


def test_sans_attente_hors_generation(tmp_path, monkeypatch):
    voice, _ = _kaggle(tmp_path, monkeypatch, installe=False)
    assert voice.moteur_voix("de") == "xtts"  # aperçus, réglages : on ne bloque pas l'interface
