"""Calage de la durée de chaque voix générée sur le temps disponible (FFmpeg atempo).

La voix est accélérée au plus de ×1,25 et ralentie au plus de ×0,85, pour rester
naturelle. Au-delà, le segment est signalé « trop long » : il faut alors raccourcir
la traduction (éditeur) puis régénérer uniquement ce segment. Une voix ne déborde jamais
sur la réplique suivante : ce qui dépasse encore est coupé, avec un court fondu.
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
FONDU_S = 0.08  # fondu de fin d'une voix coupée avant la réplique suivante
DUREE_REELLE_XTTS = 1.4  # une phrase XTTS normale dure environ 1,4 fois l'estimation (médiane mesurée)


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


def _filtrer(entree: Path, sortie: Path, filtres: list[str], seg: dict[str, Any]) -> None:
    if not filtres:
        sortie.write_bytes(entree.read_bytes())
        return
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(entree), "-filter:a", ",".join(filtres), str(sortie)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise RuntimeError(f"Calage impossible du segment {seg['id']} : {resultat.stderr.strip()}")


def caler(brut: Path, cale: Path, seg: dict[str, Any], gain_db: float = 0.0, egaliseur: str | None = None,
          niveau_cible_db: float | None = None, langue: str | None = None) -> dict[str, Any]:
    """Écrit la voix calée et renvoie {facteur, duree, trop_long, gain_db}.

    Durée (atempo), timbre (`egaliseur`, filtre FFmpeg) et volume. Avec `niveau_cible_db` (niveau de
    la phrase d'origine, pipeline/niveau.py), le volume est calé au décibel près : on mesure la voix
    après durée et timbre, puis on applique l'écart exact. Sinon, `gain_db` (écart relatif) est appliqué.
    """
    from pipeline.niveau import gain_pour_egaler, niveau_fichier_db

    from pipeline.segments import babille, duree_voix_estimee

    duree = duree_wav(brut)
    facteur, trop_long = facteur_calage(duree, seg)
    if facteur > 1.0 and langue and seg.get("traduction") and babille(duree, seg["traduction"], langue):
        # Voix trop longue à cause de sons ajoutés par XTTS après la phrase : on n'accélère que ce dont la
        # vraie phrase a besoin (durée estimée d'après son texte), le surplus est coupé ci-dessous
        phrase = DUREE_REELLE_XTTS * duree_voix_estimee(seg["traduction"], langue)
        facteur, _ = facteur_calage(min(phrase, duree), seg)
        facteur = max(facteur, 1.0)
    filtres = []
    if abs(facteur - 1.0) >= 0.01:
        filtres.append(f"atempo={facteur:.4f}")
    # Jamais sur la réplique suivante : deux voix en même temps s'entendent comme un écho. Le surplus
    # (souvent des sons ajoutés par XTTS après la phrase) est coupé, avec un court fondu
    disponible = max(seg.get("fin_max", seg["fin"]) - seg["debut"], 0.1)
    coupee = duree / facteur > disponible + 0.02
    if coupee:
        filtres += [f"atrim=end={disponible:.3f}", f"afade=t=out:st={max(disponible - FONDU_S, 0):.3f}:d={FONDU_S}"]
    if egaliseur:
        filtres.append(egaliseur)
    if niveau_cible_db is not None:
        intermediaire = cale.with_name(cale.stem + "_tmp.wav")
        _filtrer(brut, intermediaire, filtres, seg)
        gain = gain_pour_egaler(niveau_cible_db, niveau_fichier_db(intermediaire))
        gain_db = gain if gain is not None else gain_db
        source, filtres = intermediaire, []
    else:
        source = brut
    if abs(gain_db) >= 0.1:
        filtres += [f"volume={gain_db:.2f}dB", "alimiter=limit=0.95:level=false"]
    _filtrer(source, cale, filtres, seg)
    if source is not brut:
        source.unlink(missing_ok=True)
    return {"facteur": round(facteur, 3), "duree": round(min(duree / facteur, disponible), 2),
            "trop_long": trop_long, "gain_db": round(gain_db, 2), "coupee": coupee}
