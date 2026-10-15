"""Temps disponible pour chaque phrase doublée, et découpage des phrases trop longues.

Une phrase traduite est « trop longue » quand, même accélérée au maximum, la voix
doublée ne tient pas dans le temps dont elle dispose : sa durée d'origine plus le
silence qui la sépare de la phrase suivante.
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.translate import compter_syllabes, debit

ACCELERATION_MAX = 1.25  # atempo maximal accepté pour la voix doublée (phase 4)
MARGE_S = 0.1  # respiration conservée avant la phrase suivante

# Découpage : on ne crée pas de morceaux trop courts pour être doublés naturellement
DUREE_MIN_MORCEAU_S = 1.0
DUREE_CIBLE_MORCEAU_S = 2.5
PAUSE_MIN_S = 0.25
PONCTUATION_COUPURE = re.compile(r"[,;:.!?…，、。！？]$")


def marquer_fin_max(segments: list[dict[str, Any]], duree_video: float | None) -> None:
    """Ajoute à chaque segment `fin_max` : jusqu'où la voix doublée peut aller (silence suivant compris)."""
    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            limite = segments[i + 1]["debut"] - MARGE_S
        else:
            limite = duree_video if duree_video else seg["fin"]
        seg["fin_max"] = round(max(seg["fin"], limite), 2)


def ratio_requis(seg: dict[str, Any], traduction: str, source: str | None, cible: str | None,
                 avec_silence: bool = True) -> float:
    """Accélération nécessaire pour que la traduction tienne (1,0 = même vitesse que l'original)."""
    duree_originale = max(seg["fin"] - seg["debut"], 0.1)
    rapport = (compter_syllabes(traduction) / debit(cible)) / (compter_syllabes(seg["texte"]) / debit(source))
    disponible = (seg.get("fin_max", seg["fin"]) if avec_silence else seg["fin"]) - seg["debut"]
    return duree_originale * rapport / max(disponible, 0.1)


def est_trop_long(seg: dict[str, Any], traduction: str, source: str | None, cible: str | None) -> bool:
    return ratio_requis(seg, traduction, source, cible) > ACCELERATION_MAX


def _texte(mots: list[dict[str, Any]]) -> str:
    texte = " ".join(m["mot"] for m in mots)
    return re.sub(r"\s+([,;:.!?…])", r"\1", texte).strip()


def _points_de_coupure(mots: list[dict[str, Any]]) -> list[int]:
    """Indices i où l'on peut couper après mots[i] : ponctuation d'abord, sinon pauses marquées."""
    debut, fin = mots[0]["debut"], mots[-1]["fin"]
    valides = [
        i for i in range(len(mots) - 1)
        if mots[i]["fin"] - debut >= DUREE_MIN_MORCEAU_S and fin - mots[i + 1]["debut"] >= DUREE_MIN_MORCEAU_S
    ]
    ponctuation = [i for i in valides if PONCTUATION_COUPURE.search(mots[i]["mot"])]
    if ponctuation:
        return ponctuation
    return [i for i in valides if mots[i + 1]["debut"] - mots[i]["fin"] >= PAUSE_MIN_S]


def decouper(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Découpe un segment en morceaux plus courts, aux pauses naturelles, d'après le minutage des mots.

    Coupe au point le plus proche du milieu, puis recommence sur chaque moitié tant
    qu'elle dure plus de DUREE_CIBLE_MORCEAU_S. Renvoie [seg] si aucune coupure n'est possible.
    """
    mots = seg.get("mots") or []
    if len(mots) < 2 or mots[-1]["fin"] - mots[0]["debut"] <= DUREE_CIBLE_MORCEAU_S:
        return [seg]
    candidats = _points_de_coupure(mots)
    if not candidats:
        return [seg]
    milieu = (mots[0]["debut"] + mots[-1]["fin"]) / 2
    coupure = min(candidats, key=lambda i: abs(mots[i]["fin"] - milieu))
    morceaux = []
    for partie in (mots[: coupure + 1], mots[coupure + 1:]):
        morceau = {
            "debut": partie[0]["debut"],
            "fin": partie[-1]["fin"],
            "texte": _texte(partie),
            "mots": partie,
        }
        morceaux.extend(decouper(morceau))
    # Les bornes extérieures restent celles du segment d'origine
    morceaux[0]["debut"] = seg["debut"]
    morceaux[-1]["fin"] = seg["fin"]
    return morceaux
