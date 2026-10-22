"""Calibrage du débit de la voix doublée, appris sur les voix réellement générées.

La durée d'une phrase doublée est estimée à partir de son nombre de syllabes (pipeline/segments.py).
Le débit réel dépend pourtant du moteur, de la voix clonée et de la langue : mesuré sur de vrais
projets, il est souvent 15 à 40 % plus lent que l'estimation. Après chaque génération de voix, le
rapport « durée réelle / durée estimée » de chaque phrase est mesuré, et sa médiane est mémorisée par
langue (moyenne glissante sur les projets). Toutes les estimations (phrase trop longue, trop courte,
vitesse demandée au moteur, alertes de l'éditeur) l'utilisent ensuite.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("voxlingo.calibrage")

BORNES = (0.6, 1.8)  # au-delà, la mesure est aberrante (voix ratée, phrase d'un mot…)
MESURES_MIN = 3  # phrases mesurées au minimum dans un projet pour qu'il compte
POIDS_MAX = 20  # la moyenne glissante garde de la souplesse : un nouveau moteur est vite appris
SYLLABES_MIN = 6  # phrases trop courtes : durée dominée par les silences d'attaque et de fin

_verrou = threading.Lock()
_cache: dict[str, Any] = {"mtime": None, "valeurs": {}}


def _fichier() -> Path:
    from config import get_settings

    return get_settings().data_dir / "calibrage_debit.json"


def _lire() -> dict[str, dict[str, float]]:
    chemin = _fichier()
    try:
        mtime = chemin.stat().st_mtime
    except OSError:
        return {}
    if _cache["mtime"] != mtime:
        try:
            _cache["valeurs"] = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _cache["valeurs"] = {}
        _cache["mtime"] = mtime
    return _cache["valeurs"]


def facteur(langue: str | None) -> float:
    """Durée réelle / durée estimée pour cette langue (1,0 tant que rien n'a été mesuré)."""
    try:
        return float((_lire().get(langue or "") or {}).get("facteur", 1.0))
    except (TypeError, ValueError):
        return 1.0


def apprendre(langue: str, rapports: list[float]) -> float | None:
    """Ajoute les mesures d'un projet (rapports durée réelle / estimée de chaque phrase) ; renvoie le
    nouveau facteur, ou None si les mesures ne suffisent pas."""
    valides = [r for r in rapports if BORNES[0] <= r <= BORNES[1]]
    if len(valides) < MESURES_MIN:
        return None
    mesure = statistics.median(valides)
    with _verrou:
        valeurs = dict(_lire())
        avant = valeurs.get(langue) or {"facteur": 1.0, "poids": 0}
        poids = min(int(avant.get("poids", 0)), POIDS_MAX)
        nouveau = (float(avant.get("facteur", 1.0)) * poids + mesure * len(valides)) / (poids + len(valides))
        valeurs[langue] = {"facteur": round(nouveau, 3), "poids": min(poids + len(valides), POIDS_MAX)}
        chemin = _fichier()
        chemin.parent.mkdir(parents=True, exist_ok=True)
        temporaire = chemin.with_suffix(".tmp")
        temporaire.write_text(json.dumps(valeurs, ensure_ascii=False, indent=1), encoding="utf-8")
        temporaire.replace(chemin)
    logger.info("Débit des voix (%s) : mesuré ×%.2f sur %d phrases, facteur retenu ×%.2f",
                langue, mesure, len(valides), nouveau)
    return nouveau


def mesurer_projet(segments: list[dict[str, Any]], langue: str, dossier_voix: Path) -> list[float]:
    """Rapports durée réelle / durée estimée des voix générées et à jour de ce projet."""
    from pipeline.align import duree_wav
    from pipeline.segments import duree_voix_brute
    from pipeline.translate import compter_syllabes
    from pipeline.voice import Manifeste, fichier_brut

    manifeste = Manifeste(dossier_voix).donnees
    rapports = []
    for seg in segments:
        entree = manifeste.get(str(seg["id"])) or {}
        fichier = fichier_brut(dossier_voix, seg["id"])
        texte = seg.get("traduction") or ""
        if entree.get("texte") != texte or not fichier.is_file() or compter_syllabes(texte) < SYLLABES_MIN:
            continue
        vitesse = float(entree.get("vitesse") or 1.0)  # la voix a pu être demandée plus rapide
        rapports.append(duree_wav(fichier) * vitesse / max(duree_voix_brute(texte, langue), 0.1))
    return rapports
