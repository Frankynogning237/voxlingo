"""Ajustement automatique des phrases traduites trop longues.

Pour chaque phrase dont la voix doublée ne tiendrait pas dans son temps disponible :
  1. le silence qui suit la phrase est déjà compté dans ce temps (pipeline/segments.py) ;
  2. on demande à l'IA une version plus courte de la traduction ;
  3. si cela ne suffit pas, on redécoupe la phrase originale aux pauses naturelles
     et on traduit chaque morceau, calé sur son propre moment.
Les phrases qu'aucune solution ne règle restent signalées pour une correction à la main.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx

from config import get_runtime, get_settings
from pipeline.segments import (
    capacite,
    decouper,
    est_trop_long,
    marquer_fin_max,
    ratio_requis,
)
from pipeline.translate import (
    TranslationError,
    _decharger_ollama,
    _traduire_segment,
    compter_syllabes,
    ollama_disponible,
    raccourcir,
)

logger = logging.getLogger("voxlingo.adjust")

ESSAIS_REFORMULER = 3  # « reformuler » : la voix naturelle demande des phrases plus courtes


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


def ajuster_segments(
    data: dict[str, Any],
    source: str,
    cible: str,
    ids: set[int] | None = None,
    on_progress: Callable[[float], None] | None = None,
    strategie: str | None = None,
) -> dict[str, int]:
    """Ajuste les phrases trop longues de `data["segments"]` (toutes, ou seulement `ids`).

    `strategie` (« reformuler » ou « accelerer ») fixe à partir de quand une phrase est trop
    longue : dès qu'il faudrait accélérer la voix, ou seulement au-delà de l'accélération maximale.

    Modifie `data` sur place et renvoie le bilan : {silence, raccourcies, decoupees, a_revoir}.
    """
    segments: list[dict[str, Any]] = data["segments"]
    marquer_fin_max(segments, data.get("duree_s"))
    bilan = {"silence": 0, "raccourcies": 0, "decoupees": 0, "a_revoir": 0}

    # Phrases qui tiennent grâce au silence suivant (aucune modification nécessaire)
    for seg in segments:
        if (ids is None or seg["id"] in ids) and seg.get("traduction"):
            hors_silence = ratio_requis(seg, seg["traduction"], source, cible, avec_silence=False)
            if hors_silence > capacite(strategie) and not est_trop_long(seg, seg["traduction"], source, cible, strategie):
                bilan["silence"] += 1

    a_traiter = [
        i for i, seg in enumerate(segments)
        if (ids is None or seg["id"] in ids)
        and seg.get("traduction")
        and est_trop_long(seg, seg["traduction"], source, cible, strategie)
    ]
    if not a_traiter:
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
                if i not in a_traiter:
                    resultat.append(seg)
                    continue

                # 2. Raccourcir la traduction
                courte = _raccourcir(client, modele, seg, seg["traduction"], source, cible, strategie)
                if courte is not None:
                    seg["traduction"] = courte
                    seg["ajustement"] = "raccourcie"
                    bilan["raccourcies"] += 1
                    resultat.append(seg)
                else:
                    # 3. Redécouper la phrase originale et traduire chaque morceau
                    morceaux = decouper(seg)
                    if len(morceaux) > 1:
                        marquer_fin_max(morceaux, None)
                        morceaux[-1]["fin_max"] = seg.get("fin_max", seg["fin"])
                        traduits: list[str] = []
                        for k, morceau in enumerate(morceaux):
                            texte = _traduire_segment(client, modele, morceaux, k, source, cible, traduits)
                            if est_trop_long(morceau, texte, source, cible, strategie):
                                texte = _raccourcir(client, modele, morceau, texte, source, cible, strategie) or texte
                            traduits.append(texte)
                            morceau["traduction"] = texte
                            morceau["traduction_ia"] = texte
                            morceau["ajustement"] = "decoupee"
                            if seg.get("locuteur"):
                                morceau["locuteur"] = seg["locuteur"]
                        resultat.extend(morceaux)
                        bilan["decoupees"] += 1
                    else:
                        seg["ajustement"] = "a_revoir"
                        resultat.append(seg)

                fait += 1
                if on_progress:
                    on_progress(fait / len(a_traiter))
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
