"""Analyse de la prosodie (ton) de chaque phrase d'origine, pour un doublage expressif.

Pour chaque phrase : énergie (dB), hauteur médiane et variation de la voix (f0), débit
(syllabes/s). On en déduit un ton parmi « énergique », « calme », « intense », « doux »
et « neutre », relatif à la façon de parler habituelle du même locuteur.

Ce ton règle ensuite la génération :
  - l'expressivité de XTTS (température : plus haute pour un ton énergique) ;
  - le volume de la voix doublée, calé sur l'énergie de la phrase d'origine.

Il s'agit d'une analyse acoustique, pas d'une reconnaissance d'émotions : elle décrit
comment la phrase est dite (fort, doux, animé…), pas ce que ressent la personne. Le ton
se corrige dans l'éditeur.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.translate import compter_syllabes

logger = logging.getLogger("voxlingo.prosody")

TONS = ("energique", "calme", "intense", "doux", "neutre")
TEMPERATURE = {"energique": 0.85, "intense": 0.8, "neutre": 0.7, "calme": 0.6, "doux": 0.55}
GAIN_MAX_DB = 9.0  # écart de volume maximal appliqué à la voix doublée


def _mesurer(signal: np.ndarray, taux: int, texte: str, duree: float) -> dict[str, float | None]:
    import librosa

    rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
    energie = round(20 * np.log10(max(rms, 1e-5)), 1)
    f0, voise, _ = librosa.pyin(signal, fmin=60, fmax=500, sr=taux, frame_length=1024) if signal.size > taux * 0.3 else (None, None, None)
    valeurs = f0[voise & ~np.isnan(f0)] if f0 is not None else np.array([])
    hauteur = float(np.median(valeurs)) if valeurs.size >= 5 else None
    # Variation de la hauteur en demi-tons : voix monotone ou animée
    variation = float(np.std(12 * np.log2(valeurs / np.median(valeurs)))) if valeurs.size >= 5 else None
    return {
        "energie_db": energie,
        "f0_hz": round(hauteur, 1) if hauteur else None,
        "variation_dt": round(variation, 2) if variation is not None else None,
        "debit": round(compter_syllabes(texte) / max(duree, 0.1), 1),
    }


def _ton(mesure: dict[str, Any], reference: dict[str, float]) -> str:
    """Ton d'une phrase par rapport aux habitudes du même locuteur."""
    ecart_energie = mesure["energie_db"] - reference["energie_db"]
    variation = mesure.get("variation_dt")
    animee = variation is not None and variation > reference["variation_dt"] * 1.3
    monotone = variation is not None and variation < reference["variation_dt"] * 0.7
    rapide = mesure["debit"] > reference["debit"] * 1.2
    if ecart_energie > 5:
        return "intense"
    if ecart_energie < -6:
        return "doux"
    if animee or (ecart_energie > 2 and rapide):
        return "energique"
    if monotone or (ecart_energie < -2 and not rapide):
        return "calme"
    return "neutre"


def analyser(segments: list[dict[str, Any]], audio_16k: Path,
             on_progress: Callable[[float], None] | None = None) -> None:
    """Ajoute `prosodie` (mesures) et `ton` à chaque segment (le ton corrigé par l'utilisateur est conservé)."""
    import soundfile as sf

    with sf.SoundFile(str(audio_16k)) as fichier:
        taux = fichier.samplerate
        for n, seg in enumerate(segments):
            fichier.seek(int(seg["debut"] * taux))
            signal = fichier.read(int((seg["fin"] - seg["debut"]) * taux), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            seg["prosodie"] = _mesurer(signal, taux, seg["texte"], seg["fin"] - seg["debut"])
            if on_progress:
                on_progress(0.9 * (n + 1) / max(len(segments), 1))

    # Référence de chaque locuteur : sa façon habituelle de parler (médianes)
    par_locuteur: dict[str, list[dict[str, Any]]] = {}
    for seg in segments:
        par_locuteur.setdefault(seg.get("locuteur") or "SPEAKER_00", []).append(seg["prosodie"])
    references = {}
    for code, mesures in par_locuteur.items():
        variations = [m["variation_dt"] for m in mesures if m.get("variation_dt") is not None]
        references[code] = {
            "energie_db": float(np.median([m["energie_db"] for m in mesures])),
            "variation_dt": float(np.median(variations)) if variations else 1.0,
            "debit": float(np.median([m["debit"] for m in mesures])),
        }
    for seg in segments:
        if not seg.get("ton_corrige"):
            seg["ton"] = _ton(seg["prosodie"], references[seg.get("locuteur") or "SPEAKER_00"])
    if on_progress:
        on_progress(1.0)
    compte = {t: sum(1 for s in segments if s.get("ton") == t) for t in TONS}
    logger.info("Tons détectés : %s", compte)


def gain_volume(seg: dict[str, Any]) -> float:
    """Gain (dB) pour que la voix doublée suive l'énergie relative de la phrase d'origine.

    On compare l'écart de la phrase à la moyenne de son locuteur, pas des niveaux absolus :
    une phrase criée reste plus forte que les autres, une phrase chuchotée plus douce.
    """
    ecart = seg.get("prosodie", {}).get("ecart_energie_db")
    if ecart is None:
        return 0.0
    return float(np.clip(ecart, -GAIN_MAX_DB, GAIN_MAX_DB))


def marquer_ecarts(segments: list[dict[str, Any]]) -> None:
    """Écart d'énergie de chaque phrase à la médiane de son locuteur (sert à gain_volume)."""
    par_locuteur: dict[str, list[float]] = {}
    for seg in segments:
        if "prosodie" in seg:
            par_locuteur.setdefault(seg.get("locuteur") or "SPEAKER_00", []).append(seg["prosodie"]["energie_db"])
    medianes = {code: float(np.median(v)) for code, v in par_locuteur.items()}
    for seg in segments:
        if "prosodie" in seg:
            code = seg.get("locuteur") or "SPEAKER_00"
            seg["prosodie"]["ecart_energie_db"] = round(seg["prosodie"]["energie_db"] - medianes[code], 1)
