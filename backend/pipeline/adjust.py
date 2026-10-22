"""Ajustement automatique de la durée des phrases traduites.

Pour chaque phrase dont la voix doublée ne tiendrait pas dans son temps disponible :
  1. le silence qui suit la phrase est déjà compté dans ce temps (pipeline/segments.py) ;
  2. on demande à l'IA une version plus courte de la traduction ;
  3. si cela ne suffit pas, on redécoupe la phrase originale aux pauses naturelles
     et on traduit chaque morceau, calé sur son propre moment.
Les phrases qu'aucune solution ne règle restent signalées pour une correction à la main.

Pour les phrases qui tiennent, mais dont la durée ne correspond pas à la scène :
  - une phrase longue (scène de plusieurs secondes) est calée morceau par morceau sur les pauses
    de la voix d'origine : la voix doublée repart avec le personnage après chaque pause, au lieu
    d'être étirée ou compressée d'un bloc et de dériver au fil de la scène ;
  - une traduction trop courte (la voix s'arrêterait alors que le personnage parle encore) est
    reformulée un peu plus longuement par l'IA, dans le même sens.
Une traduction corrigée à la main n'est jamais modifiée par ces deux ajustements, sauf si l'on
demande l'ajustement de cette phrase précisément.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx

from config import get_runtime, get_settings
from pipeline.segments import (
    REMPLISSAGE_CIBLE,
    besoin_ajustement,
    capacite,
    decouper,
    est_trop_court,
    est_trop_long,
    marquer_fin_max,
    ratio_requis,
    syllabes_pour_duree,
)
from pipeline.translate import (
    TranslationError,
    _decharger_ollama,
    _traduire_segment,
    allonger,
    compter_syllabes,
    ollama_disponible,
    raccourcir,
    traduction_suspecte,
)

logger = logging.getLogger("voxlingo.adjust")

ESSAIS_REFORMULER = 3  # « reformuler » : la voix naturelle demande des phrases plus courtes
ESSAIS_ALLONGER = 2
# Ce que chaque morceau d'une phrase découpée garde de la phrase d'origine
HERITAGE = ("locuteur", "ton", "ton_corrige", "emotion")


def _syllabes_max(seg: dict[str, Any], traduction: str, source: str, cible: str, strategie: str) -> int:
    """Syllabes maximales pour tenir dans le temps disponible, avec une petite marge."""
    ratio = ratio_requis(seg, traduction, source, cible)
    return max(1, int(compter_syllabes(traduction) * capacite(strategie) / ratio * 0.95))


def _raccourcir(client: httpx.Client, modele: str, seg: dict[str, Any], traduction: str,
                source: str, cible: str, strategie: str) -> str | None:
    """Fait reformuler la traduction plus brièvement, dans le même sens ; renvoie la version qui tient, ou None."""
    for _ in range(ESSAIS_REFORMULER):
        courte = raccourcir(client, modele, seg["texte"], traduction, source, cible,
                            _syllabes_max(seg, traduction, source, cible, strategie))
        if not est_trop_long(seg, courte, source, cible, strategie):
            return courte
        if compter_syllabes(courte) < compter_syllabes(traduction):
            traduction = courte  # on repart de la version la plus courte obtenue
    return None


def _allonger(client: httpx.Client, modele: str, seg: dict[str, Any], traduction: str,
              source: str, cible: str, strategie: str) -> str | None:
    """Fait reformuler une traduction trop courte un peu plus longuement, dans le même sens.

    Renvoie la meilleure version obtenue (plus longue, qui tient toujours dans son temps et reste une
    traduction plausible de l'original), ou None si aucune n'améliore la phrase."""
    cible_syllabes = syllabes_pour_duree((seg["fin"] - seg["debut"]) * REMPLISSAGE_CIBLE, cible)
    meilleure, syllabes = None, compter_syllabes(traduction)
    for _ in range(ESSAIS_ALLONGER):
        longue = allonger(client, modele, seg["texte"], meilleure or traduction, source, cible,
                          max(syllabes + 1, round(cible_syllabes * 0.9)), round(cible_syllabes * 1.05))
        if (compter_syllabes(longue) <= syllabes or est_trop_long(seg, longue, source, cible, strategie)
                or traduction_suspecte(seg["texte"], longue, source, cible)):
            continue
        meilleure, syllabes = longue, compter_syllabes(longue)
        if not est_trop_court(seg, longue, cible):
            break
    return meilleure


def _traduire_morceaux(client: httpx.Client, modele: str, seg: dict[str, Any], morceaux: list[dict[str, Any]],
                       source: str, cible: str, strategie: str, ajustement: str) -> list[dict[str, Any]]:
    """Traduit chaque morceau d'une phrase découpée, calé sur son propre moment."""
    marquer_fin_max(morceaux, None)
    morceaux[-1]["fin_max"] = seg.get("fin_max", seg["fin"])
    traduits: list[str] = []
    for k, morceau in enumerate(morceaux):
        for cle in HERITAGE:
            if cle in seg:
                morceau[cle] = seg[cle]
        texte = _traduire_segment(client, modele, morceaux, k, source, cible, traduits)
        if est_trop_long(morceau, texte, source, cible, strategie):
            texte = _raccourcir(client, modele, morceau, texte, source, cible, strategie) or texte
        traduits.append(texte)
        morceau["traduction"] = texte
        morceau["traduction_ia"] = texte
        morceau["ajustement"] = ajustement
    return morceaux


def _a_traiter(seg: dict[str, Any], ids: set[int] | None, source: str, cible: str, strategie: str | None) -> str | None:
    """Ce qu'il faut faire de cette phrase : « long » (trop longue), « caler », « allonger » ou None."""
    if (ids is not None and seg["id"] not in ids) or not seg.get("traduction"):
        return None
    if est_trop_long(seg, seg["traduction"], source, cible, strategie):
        return "long"
    # Une traduction corrigée à la main n'est retouchée que si l'on demande cette phrase précisément
    if ids is None and seg["traduction"] != seg.get("traduction_ia", seg["traduction"]):
        return None
    return besoin_ajustement(seg, cible)


def a_ajuster(segments: list[dict[str, Any]], source: str, cible: str, strategie: str | None,
              ids: set[int] | None = None) -> bool:
    """Au moins une phrase a besoin d'un ajustement de durée (trop longue, longue à caler, trop courte)."""
    return any(_a_traiter(seg, ids, source, cible, strategie) for seg in segments)


def ajuster_segments(
    data: dict[str, Any],
    source: str,
    cible: str,
    ids: set[int] | None = None,
    on_progress: Callable[[float], None] | None = None,
    strategie: str | None = None,
) -> dict[str, int]:
    """Ajuste la durée des phrases de `data["segments"]` (toutes, ou seulement `ids`).

    `strategie` (« reformuler » ou « accelerer ») fixe à partir de quand une phrase est trop
    longue : dès qu'il faudrait accélérer la voix, ou seulement au-delà de l'accélération maximale.

    Modifie `data` sur place et renvoie le bilan :
    {silence, raccourcies, decoupees, calees, allongees, a_revoir}.
    """
    segments: list[dict[str, Any]] = data["segments"]
    marquer_fin_max(segments, data.get("duree_s"))
    bilan = {"silence": 0, "raccourcies": 0, "decoupees": 0, "calees": 0, "allongees": 0, "a_revoir": 0}

    # Phrases qui tiennent grâce au silence suivant (aucune modification nécessaire)
    for seg in segments:
        if (ids is None or seg["id"] in ids) and seg.get("traduction"):
            hors_silence = ratio_requis(seg, seg["traduction"], source, cible, avec_silence=False)
            if hors_silence > capacite(strategie) and not est_trop_long(seg, seg["traduction"], source, cible, strategie):
                bilan["silence"] += 1

    actions = {i: a for i, seg in enumerate(segments) if (a := _a_traiter(seg, ids, source, cible, strategie))}
    if not actions:
        return bilan

    modele = get_runtime().ollama_model
    if not ollama_disponible(modele):
        raise TranslationError(
            "L'ajustement automatique a besoin d'Ollama : lancez l'application Ollama puis réessayez."
        )

    resultat: list[dict[str, Any]] = []
    with httpx.Client(base_url=get_settings().ollama_url, timeout=180) as client:
        try:
            fait = 0
            for i, seg in enumerate(segments):
                action = actions.get(i)
                if action is None:
                    resultat.append(seg)
                    continue

                if action == "long":
                    # 2. Raccourcir la traduction
                    courte = _raccourcir(client, modele, seg, seg["traduction"], source, cible, strategie)
                    if courte is not None:
                        seg["traduction"] = courte
                        seg["ajustement"] = "raccourcie"
                        bilan["raccourcies"] += 1
                        pieces = [seg]
                    else:
                        # 3. Redécouper la phrase originale et traduire chaque morceau
                        morceaux = decouper(seg)
                        if len(morceaux) > 1:
                            pieces = _traduire_morceaux(client, modele, seg, morceaux, source, cible, strategie,
                                                        "decoupee")
                            bilan["decoupees"] += 1
                        else:
                            seg["ajustement"] = "a_revoir"
                            pieces = [seg]
                elif action == "caler":
                    # Phrase longue : un morceau par passage entre deux pauses de la voix d'origine
                    pieces = _traduire_morceaux(client, modele, seg, decouper(seg), source, cible, strategie, "calee")
                    bilan["calees"] += 1
                else:
                    pieces = [seg]

                # Morceau ou phrase trop courte pour son moment : version un peu plus développée
                for piece in pieces:
                    if piece.get("ajustement") == "a_revoir" or not est_trop_court(piece, piece["traduction"], cible):
                        continue
                    longue = _allonger(client, modele, piece, piece["traduction"], source, cible, strategie)
                    if longue is None:
                        piece["allongement_essaye"] = True  # pas de nouvel essai automatique
                        continue
                    piece["traduction"] = longue
                    if piece.get("ajustement") in ("calee", "decoupee"):
                        piece["traduction_ia"] = longue
                    else:
                        piece["ajustement"] = "allongee"
                        bilan["allongees"] += 1
                resultat.extend(pieces)

                fait += 1
                if on_progress:
                    on_progress(fait / len(actions))
        finally:
            _decharger_ollama(modele)

    # Renumérotation, temps disponibles et alertes à jour
    for n, seg in enumerate(resultat):
        seg["id"] = n
    marquer_fin_max(resultat, data.get("duree_s"))
    for seg in resultat:
        seg["trop_long"] = est_trop_long(seg, seg["traduction"], source, cible, strategie)
    bilan["a_revoir"] = sum(1 for seg in resultat if seg["trop_long"])
    data["segments"] = resultat
    logger.info("Ajustement : %s", bilan)
    return bilan
