"""Acoustique de la pièce : la voix doublée reçoit la réverbération mesurée sur la voix d'origine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from pipeline import acoustique as ac

TAUX = 24000


def _phrases(duree: float = 20.0, graine: int = 1) -> np.ndarray:
    """Parole simulée : rafales de bruit filtré de 1 s, séparées de 1 s de silence."""
    rng = np.random.default_rng(graine)
    x = np.zeros(int(duree * TAUX), dtype=np.float32)
    for debut in np.arange(0.5, duree - 1.5, 2.0):
        a = int(debut * TAUX)
        x[a:a + TAUX] = 0.2 * rng.standard_normal(TAUX)
    return x


def _ecrire(chemin: Path, x: np.ndarray) -> Path:
    sf.write(chemin, x, TAUX, subtype="FLOAT")
    return chemin


def test_piece_trouvee_quand_la_voix_d_origine_resonne(tmp_path: Path) -> None:
    ri = ac.reponse_impulsionnelle(0.45, -16, TAUX, graine=3)
    origine = fftconvolve(_phrases(), ri)[: 20 * TAUX].astype(np.float32)
    origine += 1e-5 * np.random.default_rng(2).standard_normal(len(origine)).astype(np.float32)  # léger bruit
    piece = ac.choisir_piece(_ecrire(tmp_path / "o.wav", origine), _ecrire(tmp_path / "d.wav", _phrases(graine=5)))
    assert piece is not None
    assert piece["ecart"] < piece["ecart_avant"] - ac.GAIN_MIN_DB
    assert 0.3 <= piece["reverberation_s"] <= 0.6


def test_aucune_piece_quand_la_voix_d_origine_est_seche(tmp_path: Path) -> None:
    origine = _ecrire(tmp_path / "o.wav", _phrases())
    assert ac.choisir_piece(origine, _ecrire(tmp_path / "d.wav", _phrases(graine=5))) is None


def test_aucune_piece_quand_le_bruit_masque_la_reverberation(tmp_path: Path) -> None:
    # Bruit de fond fort dans la voix d'origine : la réverbération n'y est pas mesurable, on n'invente rien
    ri = ac.reponse_impulsionnelle(0.8, -12, TAUX)
    origine = fftconvolve(_phrases(), ri)[: 20 * TAUX].astype(np.float32)
    origine += 0.02 * np.random.default_rng(2).standard_normal(len(origine)).astype(np.float32)
    assert ac.choisir_piece(_ecrire(tmp_path / "o.wav", origine), _ecrire(tmp_path / "d.wav", _phrases(graine=5))) is None


def test_application_par_blocs_identique_a_une_convolution_d_un_coup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ac, "BLOC_S", 0.7)  # blocs plus courts que la réverbération
    x = _phrases(6.0)
    sortie = ac.appliquer_piece(_ecrire(tmp_path / "e.wav", x), tmp_path / "s.wav", 1.0, -12)
    y, _ = sf.read(sortie, dtype="float32")
    attendu = fftconvolve(x, ac.reponse_impulsionnelle(1.0, -12, TAUX))[: len(x)]
    assert len(y) == len(x)
    assert np.max(np.abs(y - attendu)) < 2e-3  # écart d'arrondi du PCM 16 bits


def test_volume_d_ensemble_inchange(tmp_path: Path) -> None:
    x = _phrases(10.0)
    y, _ = sf.read(ac.appliquer_piece(_ecrire(tmp_path / "e.wav", x), tmp_path / "s.wav", 0.6, -9), dtype="float32")
    assert 10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2)) == pytest.approx(0.0, abs=0.5)
