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

# XTTS v2 parle plus lentement qu'un humain : 4,6 syllabes/s mesurées en allemand, contre
# 5,97 en moyenne pour la langue, soit ~0,77. On demande donc à XTTS de parler plus vite
# (paramètre speed, jusqu'à ×1,35), puis atempo complète jusqu'à ×1,25 : capacité ×1,69.
FACTEUR_DEBIT_XTTS = 0.77
VITESSE_XTTS_MAX = 1.35

# Découpage : on ne crée pas de morceaux trop courts pour être doublés naturellement
DUREE_MIN_MORCEAU_S = 1.0
DUREE_CIBLE_MORCEAU_S = 2.5
PAUSE_MIN_S = 0.25
# Pause longue : le personnage se tait vraiment ; on y coupe toujours, même pour un morceau d'un seul mot,
# sinon la voix doublée enchaînerait sans marquer ce silence et serait décalée pour tout le reste de la phrase
PAUSE_LONGUE_S = 0.7
PONCTUATION_COUPURE = re.compile(r"[,;:.!?…，、。！？]$")


def marquer_fin_max(segments: list[dict[str, Any]], duree_video: float | None) -> None:
    """Ajoute à chaque segment `fin_max` : jusqu'où la voix doublée peut aller (silence suivant compris)."""
    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            limite = segments[i + 1]["debut"] - MARGE_S
        else:
            limite = duree_video if duree_video else seg["fin"]
        seg["fin_max"] = round(max(seg["fin"], limite), 2)


def duree_voix_estimee(traduction: str, cible: str | None, vitesse: float = 1.0) -> float:
    """Durée estimée de la phrase lue par XTTS à la vitesse donnée."""
    return compter_syllabes(traduction) / (debit(cible) * FACTEUR_DEBIT_XTTS * vitesse)


def vitesse_xtts(seg: dict[str, Any], traduction: str, cible: str | None) -> float:
    """Vitesse de lecture à demander à XTTS pour que la phrase tienne dans son temps (1,0 à 1,35)."""
    disponible = max(seg.get("fin_max", seg["fin"]) - seg["debut"], 0.1)
    return round(min(VITESSE_XTTS_MAX, max(1.0, duree_voix_estimee(traduction, cible) / disponible)), 2)


# Stratégie quand une traduction est trop longue pour son temps (option du projet) :
#  - « reformuler » : la voix garde son débit naturel ; au-delà d'une légère tolérance,
#    l'IA reformule la phrase plus brièvement, dans le même sens ;
#  - « accelerer » : la voix est accélérée (XTTS ×1,35 puis atempo ×1,25) ; l'IA ne
#    reformule qu'au-delà de cette capacité.
STRATEGIES = ("reformuler", "accelerer")
STRATEGIE_PAR_DEFAUT = "reformuler"
TOLERANCE_REFORMULER = 1.08  # accélération imperceptible acceptée en mode « reformuler »


# Phrases longues : calées morceau par morceau sur les pauses de la voix d'origine, pour que la voix
# doublée reparte avec le personnage après chaque pause au lieu de dériver sur toute la scène
DUREE_LONGUE_S = 6.0
# Traductions trop courtes : même ralentie au maximum (×0,85), la voix laisserait le personnage parler
# sans voix à la fin de la phrase ; l'IA en propose alors une version un peu plus développée
DUREE_MIN_ALLONGER_S = 1.5
REMPLISSAGE_MIN = 0.8  # en dessous de 80 % de la durée d'origine (au débit naturel), la phrase est allongée
REMPLISSAGE_CIBLE = 0.95


def syllabes_pour_duree(duree_s: float, cible: str | None) -> int:
    """Nombre de syllabes que la voix doublée prononce en `duree_s` secondes, à son débit naturel."""
    return max(1, round(duree_s * debit(cible) * FACTEUR_DEBIT_XTTS))


def est_trop_court(seg: dict[str, Any], traduction: str, cible: str | None) -> bool:
    """La voix doublée, au débit naturel, laisserait une bonne partie de la phrase d'origine sans voix."""
    originale = seg["fin"] - seg["debut"]
    return originale >= DUREE_MIN_ALLONGER_S and duree_voix_estimee(traduction, cible) < REMPLISSAGE_MIN * originale


def a_caler(seg: dict[str, Any]) -> bool:
    """Phrase longue qui peut être découpée aux pauses de la voix d'origine."""
    return seg["fin"] - seg["debut"] >= DUREE_LONGUE_S and len(decouper(seg)) > 1


def besoin_ajustement(seg: dict[str, Any], cible: str | None) -> str | None:
    """Ajustement de durée utile à une phrase qui tient dans son temps : « caler » (phrase longue à
    découper aux pauses d'origine), « allonger » (traduction trop courte) ou None."""
    if not seg.get("traduction"):
        return None
    if a_caler(seg):
        return "caler"
    if not seg.get("allongement_essaye") and est_trop_court(seg, seg["traduction"], cible):
        return "allonger"
    return None


def capacite(strategie: str | None) -> float:
    """Accélération totale acceptée avant de considérer une phrase comme trop longue."""
    if (strategie or STRATEGIE_PAR_DEFAUT) == "accelerer":
        return VITESSE_XTTS_MAX * ACCELERATION_MAX
    return TOLERANCE_REFORMULER


def ratio_requis(seg: dict[str, Any], traduction: str, source: str | None, cible: str | None,
                 avec_silence: bool = True) -> float:
    """Accélération totale nécessaire pour que la voix tienne (1,0 = débit naturel de XTTS).

    `source` est conservé pour la compatibilité des appels : seule compte la durée de la voix
    face au temps disponible (durée d'origine, plus le silence suivant si `avec_silence`).
    """
    disponible = (seg.get("fin_max", seg["fin"]) if avec_silence else seg["fin"]) - seg["debut"]
    return duree_voix_estimee(traduction, cible) / max(disponible, 0.1)


def est_trop_long(seg: dict[str, Any], traduction: str, source: str | None, cible: str | None,
                  strategie: str | None = None) -> bool:
    return ratio_requis(seg, traduction, source, cible) > capacite(strategie)


def _texte(mots: list[dict[str, Any]]) -> str:
    texte = " ".join(m["mot"] for m in mots)
    return re.sub(r"\s+([,;:.!?…])", r"\1", texte).strip()


def _points_de_coupure(mots: list[dict[str, Any]]) -> list[int]:
    """Indices i où l'on peut couper après mots[i] : pauses longues d'abord, puis ponctuation, sinon
    pauses marquées."""
    longues = [i for i in range(len(mots) - 1) if mots[i + 1]["debut"] - mots[i]["fin"] >= PAUSE_LONGUE_S]
    if longues:
        return longues
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
