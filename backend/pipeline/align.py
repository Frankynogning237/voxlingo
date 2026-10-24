"""Calage de la durée de chaque voix générée sur le temps disponible (FFmpeg atempo).

La voix est accélérée au plus de ×1,25 et ralentie au plus de ×0,85, pour rester
naturelle. Au-delà, le segment est signalé « trop long » : il faut alors raccourcir
la traduction (éditeur) puis régénérer uniquement ce segment.

Une phrase qui contient des pauses est calée morceau par morceau sur les passages parlés d'origine
(pipeline/calage_pauses.py) : la voix se tait quand le personnage se tait. Une voix un peu plus longue
que la phrase d'origine est d'abord légèrement accélérée (×1,12 au plus, imperceptible) pour finir avec
la bouche du personnage ; le silence qui suit n'est utilisé qu'au-delà.
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
ACCELERATION_DOUCE = 1.12  # pour finir avec la phrase d'origine plutôt que de déborder dans le silence


def duree_wav(chemin: Path) -> float:
    info = sf.info(str(chemin))
    return info.frames / info.samplerate


def facteur_calage(duree_generee: float, seg: dict[str, Any]) -> tuple[float, bool]:
    """Facteur atempo à appliquer (>1 accélère) et indicateur « trop long »."""
    originale = max(seg["fin"] - seg["debut"], 0.1)
    disponible = max(seg.get("fin_max", seg["fin"]) - seg["debut"], 0.1)
    if duree_generee > disponible:
        facteur = max(duree_generee / disponible, min(duree_generee / originale, ACCELERATION_DOUCE))
        return min(facteur, ACCELERATION_MAX), duree_generee / disponible > ACCELERATION_MAX
    if duree_generee > originale:
        # Voix un peu plus longue que la phrase : légère accélération pour finir avec la bouche du personnage
        return min(duree_generee / originale, ACCELERATION_DOUCE), False
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
          niveau_cible_db: float | None = None, origine: Path | None = None,
          hauteur_cible_hz: float | None = None, hauteur_part: float = 1.0) -> dict[str, Any]:
    """Écrit la voix calée et renvoie {facteur, duree, trop_long, gain_db}.

    Durée (atempo), timbre (`egaliseur`, filtre FFmpeg) et volume. Avec `niveau_cible_db` (niveau de
    la phrase d'origine, pipeline/niveau.py), le volume est calé au décibel près : on mesure la voix
    après durée et timbre, puis on applique l'écart exact. Sinon, `gain_db` (écart relatif) est appliqué.
    `origine` (voix d'origine isolée) situe les pauses du personnage pour le calage morceau par morceau.
    `hauteur_cible_hz` (voix clonées, pipeline/hauteur.py) : la voix est transposée vers cette hauteur
    (`hauteur_part` de l'écart, choisie à l'empreinte vocale).
    """
    from pipeline.calage_pauses import caler_sur_pauses
    from pipeline.niveau import gain_pour_egaler, niveau_fichier_db

    from pipeline.hauteur import decalage, filtre, hauteur_mediane

    duree = duree_wav(brut)
    facteur, trop_long = facteur_calage(duree, seg)
    filtres = []
    demi_tons = decalage(hauteur_mediane(brut), hauteur_cible_hz, hauteur_part) if hauteur_cible_hz and hauteur_part else 0.0
    morceaux = cale.with_name(cale.stem + "_pauses.wav")
    par_pauses = caler_sur_pauses(brut, seg, morceaux, origine)
    if par_pauses is not None:  # durée déjà calée, morceau par morceau
        brut, facteur, trop_long = morceaux, par_pauses["facteur"], False
    elif abs(facteur - 1.0) >= 0.01:
        filtres.append(f"atempo={facteur:.4f}")
    if demi_tons:
        filtres.append(filtre(demi_tons))
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
    morceaux.unlink(missing_ok=True)
    duree_finale = par_pauses["duree"] if par_pauses is not None else round(duree / facteur, 2)
    return {"facteur": round(facteur, 3), "duree": duree_finale, "trop_long": trop_long,
            "gain_db": round(gain_db, 2), "cale_sur_pauses": par_pauses is not None,
            "hauteur_dt": round(demi_tons, 1)}
