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
