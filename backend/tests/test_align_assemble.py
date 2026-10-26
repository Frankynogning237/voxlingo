"""Tests de base du calage des voix (align.py) et de l'assemblage (assemble.py, segments.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.align import RALENTISSEMENT_MAX, facteur_calage
from pipeline.assemble import TAUX, construire_piste, ecrire_srt
from pipeline.segments import ACCELERATION_MAX, decouper, marquer_fin_max


def _wav(dossier: Path, nom: str, valeur: float, duree: float) -> Path:
    chemin = dossier / f"{nom}.wav"
    sf.write(chemin, np.full(int(duree * TAUX), valeur, dtype=np.float32), TAUX, subtype="FLOAT")
    return chemin


# ───────────────────────── align.py ─────────────────────────

SEGMENT = {"id": 0, "debut": 1.0, "fin": 3.0, "fin_max": 3.5}  # 2 s de parole, 2,5 s disponibles


def test_voix_qui_tient_n_est_pas_modifiee() -> None:
    assert facteur_calage(2.2, SEGMENT) == (1.0, False)


def test_voix_trop_longue_est_acceleree_dans_le_temps_disponible() -> None:
    facteur, trop_long = facteur_calage(3.0, SEGMENT)
    assert facteur == pytest.approx(3.0 / 2.5)
    assert not trop_long


def test_acceleration_plafonnee_et_signalee() -> None:
    facteur, trop_long = facteur_calage(4.0, SEGMENT)
    assert facteur == ACCELERATION_MAX
    assert trop_long


def test_voix_trop_courte_ralentie_au_plus_de_15_pourcent() -> None:
    facteur, trop_long = facteur_calage(1.0, SEGMENT)
    assert facteur == RALENTISSEMENT_MAX
    assert not trop_long


# ───────────────────────── assemble.py ─────────────────────────

def test_piste_a_la_duree_de_la_video(tmp_path: Path) -> None:
    piste = construire_piste([(0.5, _wav(tmp_path, "a", 0.1, 1.0))], 4.0, tmp_path / "piste.wav")
    signal, taux = sf.read(piste)
    assert taux == TAUX
    assert len(signal) == 4 * TAUX


def test_voix_placees_au_bon_moment_et_chevauchements_additionnes(tmp_path: Path) -> None:
    a = _wav(tmp_path, "a", 0.1, 0.5)
    b = _wav(tmp_path, "b", 0.2, 0.5)
    c = _wav(tmp_path, "c", 0.3, 0.4)
    construire_piste([(0.0, a), (0.3, b), (2.8, c)], 3.0, tmp_path / "piste.wav")
    x, _ = sf.read(tmp_path / "piste.wav")
    echantillon = lambda t: x[int(t * TAUX)]  # noqa: E731
    assert echantillon(0.1) == pytest.approx(0.1, abs=1e-3)  # voix a seule
    assert echantillon(0.4) == pytest.approx(0.3, abs=1e-3)  # a + b se chevauchent
    assert echantillon(0.6) == pytest.approx(0.2, abs=1e-3)  # b seule
    assert echantillon(1.5) == pytest.approx(0.0, abs=1e-3)  # silence
    assert echantillon(2.9) == pytest.approx(0.3, abs=1e-3)  # c, coupée à la fin de la vidéo


def test_voix_au_dela_de_la_video_ignoree(tmp_path: Path) -> None:
    construire_piste([(5.0, _wav(tmp_path, "a", 0.5, 1.0))], 2.0, tmp_path / "piste.wav")
    x, _ = sf.read(tmp_path / "piste.wav")
    assert len(x) == 2 * TAUX
    assert np.abs(x).max() == pytest.approx(0.0)


def test_sous_titres_srt(tmp_path: Path) -> None:
    srt = ecrire_srt(
        [{"debut": 1.0, "fin": 3.0, "traduction": "Hallo zusammen", "duree_doublage": 2.25},
         {"debut": 3661.5, "fin": 3663.0, "traduction": "Tschüss"}],
        tmp_path / "t.srt",
    )
    texte = srt.read_text(encoding="utf-8")
    assert "1\n00:00:01,000 --> 00:00:03,250\nHallo zusammen" in texte
    assert "2\n01:01:01,500 --> 01:01:03,000\nTschüss" in texte


# ───────────────────────── segments.py ─────────────────────────

def test_temps_disponible_inclut_le_silence_suivant() -> None:
    segments = [{"debut": 0.0, "fin": 2.0}, {"debut": 3.0, "fin": 4.0}]
    marquer_fin_max(segments, duree_video=10.0)
    assert segments[0]["fin_max"] == pytest.approx(2.9)  # jusqu'à la phrase suivante, moins 0,1 s
    assert segments[1]["fin_max"] == pytest.approx(10.0)  # dernière phrase : jusqu'à la fin de la vidéo


def test_decoupage_aux_pauses_naturelles() -> None:
    phrase = "It listens to the speaker, writes down every sentence, and then translates it into another language."
    mots, t = [], 7.0
    for m in phrase.split():
        mots.append({"mot": m, "debut": round(t, 2), "fin": round(t + 0.32, 2)})
        t += 0.38
    morceaux = decouper({"debut": 7.0, "fin": round(t, 2), "texte": phrase, "mots": mots})
    assert [m["texte"] for m in morceaux] == [
        "It listens to the speaker,",
        "writes down every sentence,",
        "and then translates it into another language.",
    ]
    assert morceaux[0]["debut"] == 7.0
    assert morceaux[-1]["fin"] == round(t, 2)
