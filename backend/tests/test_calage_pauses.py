"""Calage de la voix doublée sur les pauses de la voix d'origine : la voix se tait quand le personnage se tait."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import calage_pauses as cp

TAUX = 16000


def _voix(passages: list[tuple[float, float]], duree: float) -> np.ndarray:
    """Signal de parole simulé : un son (200 Hz) pendant chaque passage, silence ailleurs."""
    t = np.arange(int(duree * TAUX)) / TAUX
    voix = np.zeros_like(t, dtype=np.float32)
    for a, b in passages:
        dedans = (t >= a) & (t < b)
        voix[dedans] = 0.3 * np.sin(2 * np.pi * 200 * t[dedans])
    return voix


def _ecrire(chemin: Path, voix: np.ndarray) -> Path:
    sf.write(chemin, voix, TAUX)
    return chemin


def _parle(voix: np.ndarray, a: float, b: float) -> float:
    """Part du temps [a, b] où la voix parle."""
    actif = cp.activite(voix, TAUX)
    return float(actif[int(a / cp.TRAME_S): int(b / cp.TRAME_S)].mean())


def test_passages_d_origine_regroupes_d_apres_les_mots() -> None:
    seg = {"debut": 10.0, "fin": 14.0, "mots": [
        {"mot": "a", "debut": 10.0, "fin": 10.5}, {"mot": "b", "debut": 10.6, "fin": 11.0},  # liaison courte
        {"mot": "c", "debut": 12.0, "fin": 12.8},  # après une vraie pause
    ]}
    assert cp.passages_origine(seg) == [(0.0, pytest.approx(1.0)), (pytest.approx(2.0), pytest.approx(2.8))]


def test_passages_mesures_dans_la_voix_d_origine(tmp_path: Path) -> None:
    origine = _ecrire(tmp_path / "origine.wav", _voix([(1.0, 2.0), (3.0, 4.0)], 6.0))
    passages = cp.passages_audio(origine, {"debut": 0.5, "fin": 5.0})
    assert len(passages) == 2
    assert passages[0][0] == pytest.approx(0.5, abs=0.03) and passages[1][0] == pytest.approx(2.5, abs=0.03)


def test_respirations_de_la_voix_generee() -> None:
    actif = cp.activite(_voix([(0.0, 1.0), (1.3, 2.0), (2.05, 3.0)], 3.0), TAUX)
    silences = cp.respirations(actif)
    assert len(silences) == 1  # 50 ms entre 2,0 et 2,05 s : trop court pour y couper
    assert silences[0][0] == pytest.approx(1.0, abs=0.03)


def test_appariement_respecte_les_pauses_au_meme_endroit_du_texte() -> None:
    passages = [(0.0, 2.0), (3.0, 5.0)]  # pause au milieu de la parole
    # respirations de la voix générée à 20 %, 50 % et 80 % de sa parole : celle du milieu est choisie
    assert cp.apparier(passages, [0.2, 0.5, 0.8]) == [(0, 1)]


def test_pause_trop_courte_sautee_plutot_que_mal_placee() -> None:
    passages = [(0.0, 3.8), (4.0, 5.0)]  # pause de 0,2 s vers 80 % de la parole
    assert cp.apparier(passages, [0.1]) == []  # respiration à 10 % : trop loin, la pause est sautée


def test_voix_posee_sur_les_passages_d_origine(tmp_path: Path) -> None:
    # Le personnage parle 0-2 s, se tait 2-4 s, reparle 4-6 s ; la voix générée enchaîne avec une
    # respiration de 0,3 s au milieu
    seg = {"debut": 0.0, "fin": 6.0, "fin_max": 6.5}
    origine = _ecrire(tmp_path / "origine.wav", _voix([(0.0, 2.0), (4.0, 6.0)], 7.0))
    brut = _ecrire(tmp_path / "brut.wav", _voix([(0.0, 2.0), (2.3, 4.3)], 4.3))
    resultat = cp.caler_sur_pauses(brut, seg, tmp_path / "cale.wav", origine)
    assert resultat is not None
    cale, _ = sf.read(tmp_path / "cale.wav", dtype="float32")
    assert _parle(cale, 2.3, 3.7) == 0.0  # silence pendant la pause du personnage
    assert _parle(cale, 4.2, 5.8) > 0.9  # et reprise avec lui
    assert resultat["duree"] <= 6.5


def test_voix_decalee_quand_le_personnage_commence_apres_un_silence(tmp_path: Path) -> None:
    seg = {"debut": 0.0, "fin": 5.0, "fin_max": 5.0}
    origine = _ecrire(tmp_path / "origine.wav", _voix([(2.0, 5.0)], 5.0))
    brut = _ecrire(tmp_path / "brut.wav", _voix([(0.0, 3.0)], 3.0))
    assert cp.caler_sur_pauses(brut, seg, tmp_path / "cale.wav", origine) is not None
    cale, _ = sf.read(tmp_path / "cale.wav", dtype="float32")
    assert _parle(cale, 0.0, 1.8) == 0.0 and _parle(cale, 2.2, 4.8) > 0.9


def test_sans_pause_ni_decalage_le_calage_d_un_bloc_s_applique(tmp_path: Path) -> None:
    seg = {"debut": 0.0, "fin": 3.0, "fin_max": 3.5}
    origine = _ecrire(tmp_path / "origine.wav", _voix([(0.0, 3.0)], 3.5))
    brut = _ecrire(tmp_path / "brut.wav", _voix([(0.0, 1.4), (1.7, 3.0)], 3.0))
    assert cp.caler_sur_pauses(brut, seg, tmp_path / "cale.wav", origine) is None


def test_morceaux_qui_ne_tiennent_pas_laissent_le_calage_d_un_bloc(tmp_path: Path) -> None:
    # Passages d'origine bien plus courts que la voix générée : même accélérée, elle ne tiendrait pas
    seg = {"debut": 0.0, "fin": 3.0, "fin_max": 3.0}
    origine = _ecrire(tmp_path / "origine.wav", _voix([(0.0, 1.0), (2.0, 3.0)], 3.0))
    brut = _ecrire(tmp_path / "brut.wav", _voix([(0.0, 2.5), (2.8, 5.0)], 5.0))
    assert cp.caler_sur_pauses(brut, seg, tmp_path / "cale.wav", origine) is None


def test_nuances_suivent_l_intensite_d_origine_autour_du_volume_de_la_phrase() -> None:
    # Le personnage crie sur le 2e passage (+8 dB) ; la voix générée est uniforme
    gains = cp.nuances([-30.0, -22.0], [-25.0, -25.0], [1.0, 1.0])
    assert gains == [pytest.approx(-4.0), pytest.approx(4.0)]
    assert sum(gains) == pytest.approx(0.0)  # le volume de la phrase reste calé ailleurs
    assert cp.nuances([-40.0, -10.0], [-25.0, -25.0], [1.0, 1.0]) == [-cp.NUANCE_MAX_DB, cp.NUANCE_MAX_DB]
    assert cp.nuances([None, -20.0], [-25.0, -25.0], [1.0, 1.0]) == [0.0, 0.0]


def test_morceau_crie_plus_fort_dans_la_voix_calee(tmp_path: Path) -> None:
    seg = {"debut": 0.0, "fin": 6.0, "fin_max": 6.5}
    origine = _voix([(0.0, 2.0), (4.0, 6.0)], 7.0)
    origine[int(4.0 * TAUX):] *= 2.5  # deuxième passage ~8 dB plus fort
    brut = _ecrire(tmp_path / "brut.wav", _voix([(0.0, 2.0), (2.3, 4.3)], 4.3))
    assert cp.caler_sur_pauses(brut, seg, tmp_path / "cale.wav", _ecrire(tmp_path / "o.wav", origine)) is not None
    cale, _ = sf.read(tmp_path / "cale.wav", dtype="float32")
    ecart = cp.niveau_actif(cale[int(4.2 * TAUX):int(5.8 * TAUX)], TAUX) - cp.niveau_actif(cale[:int(1.8 * TAUX)], TAUX)
    assert ecart == pytest.approx(20 * np.log10(2.5), abs=0.5)  # le cri d'origine est reproduit
