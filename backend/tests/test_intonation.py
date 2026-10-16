"""Tests de l'intonation de la phrase d'origine (voix clonées)."""

import numpy as np
import soundfile as sf

from pipeline.voice import Manifeste, extraire_intonation, fichier_brut

SEG = {"id": 3, "debut": 1.0, "fin": 3.0, "fin_max": 3.2, "traduction": "Hallo", "ton": "neutre"}


def test_extraire_intonation_bornes(tmp_path):
    source = tmp_path / "voix.wav"
    sf.write(source, np.zeros(16000 * 5, dtype=np.float32), 16000)
    assert extraire_intonation(source, SEG, tmp_path / "a.wav") == tmp_path / "a.wav"
    assert (tmp_path / "a.wav").is_file()
    assert extraire_intonation(source, {**SEG, "fin": 1.5}, tmp_path / "b.wav") is None  # trop courte


def test_manifeste_compatible_avec_les_anciens_projets(tmp_path):
    fichier = fichier_brut(tmp_path, SEG["id"])
    sf.write(fichier, np.zeros(100, dtype=np.float32), 24000)
    manifeste = Manifeste(tmp_path)
    manifeste.noter(SEG, "de", "ref:x.wav")
    # Une entrée écrite avant l'intonation (sans la clé) reste à jour : pas de régénération inutile
    del manifeste.donnees[str(SEG["id"])]["intonation"]
    assert manifeste.a_jour(SEG, "de", "ref:x.wav", fichier)
    # Activer l'intonation sur la phrase la fait refaire une fois
    assert not manifeste.a_jour({**SEG, "intonation": "phrase.wav"}, "de", "ref:x.wav", fichier)
