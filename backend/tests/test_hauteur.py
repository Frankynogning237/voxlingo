"""Hauteur de la voix doublée calée sur celle du personnage (voix clonées)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import hauteur
from pipeline.align import caler

TAUX = 24000


def _voix(f0: float, duree: float = 2.0) -> np.ndarray:
    """Voyelle simulée : harmoniques de `f0`, légère modulation d'amplitude."""
    t = np.arange(int(duree * TAUX)) / TAUX
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    return (0.15 * x * (0.8 + 0.2 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_cible_entre_la_phrase_et_le_locuteur() -> None:
    seg = {"prosodie": {"f0_hz": 400.0}}
    assert hauteur.cible(seg, 300.0) == pytest.approx(math.sqrt(400 * 300))
    assert hauteur.cible({"prosodie": {}}, 300.0) == 300.0
    assert hauteur.cible({}, None) is None


def test_decalage_seulement_si_audible_et_plafonne() -> None:
    assert hauteur.decalage(240.0, 245.0) == 0.0  # 0,36 demi-ton : inaudible
    assert hauteur.decalage(240.0, 240 * 2 ** (3 / 12)) == pytest.approx(3.0)
    assert hauteur.decalage(200.0, 400.0) == hauteur.DECALAGE_MAX_DT
    assert hauteur.decalage(None, 300.0) == 0.0
    assert hauteur.filtre(0.0) is None


def test_voix_clonee_trop_grave_remontee_a_la_hauteur_du_personnage(tmp_path: Path) -> None:
    brut = tmp_path / "brut.wav"
    sf.write(brut, _voix(220.0), TAUX)
    seg = {"id": 0, "debut": 0.0, "fin": 2.0, "fin_max": 2.5}
    resultat = caler(brut, tmp_path / "cale.wav", seg, hauteur_cible_hz=280.0)
    assert resultat["hauteur_dt"] == pytest.approx(12 * math.log2(280 / 220), abs=0.2)
    assert hauteur.hauteur_mediane(tmp_path / "cale.wav") == pytest.approx(280.0, rel=0.03)


def test_sans_cible_la_hauteur_est_gardee(tmp_path: Path) -> None:
    brut = tmp_path / "brut.wav"
    sf.write(brut, _voix(220.0), TAUX)
    resultat = caler(brut, tmp_path / "cale.wav", {"id": 0, "debut": 0.0, "fin": 2.0, "fin_max": 2.5})
    assert resultat["hauteur_dt"] == 0.0


def test_part_de_la_correction() -> None:
    # Demi-correction d'un écart de 4 demi-tons : 2 ; d'un écart plafonné (8 → 5) : 2,5 ; trop petite : rien
    assert hauteur.decalage(200.0, 200 * 2 ** (4 / 12), part=0.5) == pytest.approx(2.0)
    assert hauteur.decalage(200.0, 200 * 2 ** (8 / 12), part=0.5) == pytest.approx(2.5)
    assert hauteur.decalage(200.0, 200 * 2 ** (0.8 / 12), part=0.5) == 0.0


def test_part_nulle_aucune_transposition(tmp_path: Path) -> None:
    brut = tmp_path / "brut.wav"
    sf.write(brut, _voix(220.0), TAUX)
    seg = {"id": 0, "debut": 0.0, "fin": 2.0, "fin_max": 2.5}
    assert caler(brut, tmp_path / "cale.wav", seg, hauteur_cible_hz=280.0, hauteur_part=0.0)["hauteur_dt"] == 0.0
