"""Hauteur de la voix doublée calée sur celle du personnage.

Le clonage reproduit le timbre, mais pas toujours la hauteur : une voix d'enfant (300-400 Hz) ressort souvent
plus grave, comme une voix d'adulte, et toutes les phrases sortent à la même hauteur alors que le personnage
monte quand il s'emporte et descend quand il se calme. Pour chaque phrase d'une voix clonée :
  - cible : la hauteur médiane de la phrase d'origine, ramenée à moitié (en demi-tons) vers la hauteur
    habituelle du locuteur, pour ne pas suivre une mesure faussée (autre voix, musique) ;
  - la voix générée est transposée vers cette cible si l'écart dépasse ECART_MIN_DT, d'au plus
    DECALAGE_MAX_DT, timbre préservé (formants conservés : la voix reste celle du clone).
Une voix choisie dans la bibliothèque garde sa hauteur (choisie, pas recopiée).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ECART_MIN_DT = 0.5  # en dessous, inaudible : rien n'est fait
DECALAGE_MAX_DT = 5.0
POIDS_PHRASE = 0.5  # part de la hauteur de la phrase dans la cible (le reste : hauteur du locuteur)
TAUX_ANALYSE = 16000


def hauteur_mediane(chemin: Path) -> float | None:
    """Hauteur médiane (Hz) des passages voisés d'un fichier audio, ou None si trop peu mesurable."""
    import librosa

    x, taux = sf.read(chemin, dtype="float32", always_2d=True)
    x = librosa.resample(x.mean(axis=1), orig_sr=taux, target_sr=TAUX_ANALYSE) if taux != TAUX_ANALYSE else x.mean(axis=1)
    if len(x) < TAUX_ANALYSE * 0.3:
        return None
    f0, voise, _ = librosa.pyin(x, fmin=60, fmax=700, sr=TAUX_ANALYSE, frame_length=1024, hop_length=320)
    valeurs = f0[voise & ~np.isnan(f0)]
    return float(np.median(valeurs)) if valeurs.size >= 8 else None


def cible(seg: dict[str, Any], f0_locuteur: float | None) -> float | None:
    """Hauteur visée pour la phrase (Hz) : celle de la phrase d'origine, ramenée vers celle du locuteur."""
    phrase = (seg.get("prosodie") or {}).get("f0_hz")
    if phrase and f0_locuteur:
        return math.exp(POIDS_PHRASE * math.log(phrase) + (1 - POIDS_PHRASE) * math.log(f0_locuteur))
    return phrase or f0_locuteur or None


def decalage(mesuree: float | None, visee: float | None) -> float:
    """Transposition (demi-tons) à appliquer : 0 si inutile ou non mesurable, plafonnée sinon."""
    if not mesuree or not visee:
        return 0.0
    ecart = 12 * math.log2(visee / mesuree)
    if abs(ecart) < ECART_MIN_DT:
        return 0.0
    return max(-DECALAGE_MAX_DT, min(DECALAGE_MAX_DT, ecart))


def filtre(demi_tons: float) -> str | None:
    """Filtre FFmpeg de transposition, timbre préservé, ou None."""
    if not demi_tons:
        return None
    return f"rubberband=pitch={2 ** (demi_tons / 12):.5f}:formant=preserved:pitchq=quality"
