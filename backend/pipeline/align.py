"""Calage de la durée de chaque voix générée sur le temps disponible (FFmpeg atempo).

La voix est accélérée au plus de ×1,25 et ralentie au plus de ×0,85, pour rester
naturelle. Au-delà, le segment est signalé « trop long » : il faut alors raccourcir
la traduction (éditeur) puis régénérer uniquement ce segment.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from pipeline.extract import _require
from pipeline.segments import ACCELERATION_MAX

logger = logging.getLogger("voxlingo.align")

RALENTISSEMENT_MAX = 0.85


def duree_wav(chemin: Path) -> float:
    info = sf.info(str(chemin))
    return info.frames / info.samplerate


def facteur_calage(duree_generee: float, seg: dict[str, Any]) -> tuple[float, bool]:
    """Facteur atempo à appliquer (>1 accélère) et indicateur « trop long »."""
    originale = max(seg["fin"] - seg["debut"], 0.1)
    disponible = max(seg.get("fin_max", seg["fin"]) - seg["debut"], 0.1)
    if duree_generee > disponible:
        facteur = duree_generee / disponible
        return min(facteur, ACCELERATION_MAX), facteur > ACCELERATION_MAX
    if duree_generee < originale:
        # Voix plus courte que l'original : on la ralentit un peu pour suivre le rythme
        return max(duree_generee / originale, RALENTISSEMENT_MAX), False
    return 1.0, False


def caler(brut: Path, cale: Path, seg: dict[str, Any], gain_db: float = 0.0) -> dict[str, Any]:
    """Écrit la voix calée (durée et volume) et renvoie {facteur, duree, trop_long}."""
    duree = duree_wav(brut)
    facteur, trop_long = facteur_calage(duree, seg)
    filtres = []
    if abs(facteur - 1.0) >= 0.01:
        filtres.append(f"atempo={facteur:.4f}")
    if abs(gain_db) >= 0.5:
        filtres += [f"volume={gain_db:.1f}dB", "alimiter=limit=0.95"]
    if not filtres:
        cale.write_bytes(brut.read_bytes())
    else:
        resultat = subprocess.run(
            [_require("ffmpeg"), "-y", "-v", "error", "-i", str(brut),
             "-filter:a", ",".join(filtres), str(cale)],
            capture_output=True, text=True,
        )
        if resultat.returncode != 0:
            raise RuntimeError(f"Calage impossible du segment {seg['id']} : {resultat.stderr.strip()}")
    return {"facteur": round(facteur, 3), "duree": round(duree / facteur, 2), "trop_long": trop_long}
