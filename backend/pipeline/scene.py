"""Comprendre la vidéo avant de la traduire : ce qui se passe à l'image, et la fiche de la vidéo.

1. **Scènes** : un modèle de vision (Ollama, par défaut Qwen 2.5 VL sur GPU) regarde une image de
   chaque plan et décrit en une phrase le lieu, les personnages, l'action et l'ambiance.
2. **Fiche de la vidéo** : le modèle de langue lit toute la transcription (avec les locuteurs,
   leurs émotions et les scènes) et rédige une fiche : genre, résumé, personnages et relations,
   registre de langue, ton général. La fiche se corrige dans l'éditeur.

La traduction reçoit ensuite, pour chaque phrase, la fiche, la scène en cours, qui parle et avec
quelle émotion : elle adapte l'intention et le ton au lieu de traduire mot à mot.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.scene")

PLAN_MIN_S = 2.0  # plans plus courts regroupés avec le suivant
SCENES_MAX = 40  # au-delà, les plans voisins sont regroupés (temps de calcul borné)
TRANSCRIPTION_MAX = 9000  # caractères de transcription lus pour la fiche

PROMPT_SCENE = (
    "Voici une image extraite d'une vidéo. Décris en une seule phrase, en français : le lieu, "
    "les personnages visibles (âge apparent, tenue, attitude), ce qu'ils font et l'ambiance. "
    "Pas d'introduction, seulement la phrase."
)

SCHEMA_FICHE = {
    "type": "object",
    "properties": {
        "genre": {"type": "string"},
        "resume": {"type": "string"},
        "registre": {"type": "string"},
        "ton": {"type": "string"},
        "personnages": {
            "type": "array",
            "items": {"type": "object", "properties": {"locuteur": {"type": "string"}, "description": {"type": "string"}},
                      "required": ["locuteur", "description"]},
        },
    },
    "required": ["genre", "resume", "registre", "ton", "personnages"],
}


def _fenetres(plans_s: list[float], duree: float) -> list[tuple[float, float]]:
    """Intervalles à décrire : les plans, regroupés quand ils sont trop courts ou trop nombreux."""
    bornes = sorted({0.0, *[p for p in plans_s if 0 < p < duree]}) + [duree]
    fenetres: list[list[float]] = []
    for debut, fin in zip(bornes, bornes[1:], strict=False):
        if fenetres and (fenetres[-1][1] - fenetres[-1][0] < PLAN_MIN_S or fin - debut < PLAN_MIN_S / 2):
            fenetres[-1][1] = fin
        else:
            fenetres.append([debut, fin])
    while len(fenetres) > SCENES_MAX:  # on fusionne la plus courte avec sa voisine
        i = min(range(len(fenetres) - 1), key=lambda k: fenetres[k][1] - fenetres[k][0])
        fenetres[i][1] = fenetres[i + 1][1]
        del fenetres[i + 1]
    return [(a, b) for a, b in fenetres]


def _image(video: Path, instant: float) -> bytes | None:
    from pipeline.extract import _require

    resultat = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-ss", f"{instant:.2f}", "-i", str(video), "-frames:v", "1",
         "-vf", "scale='min(640,iw)':-2", "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "-"],
        capture_output=True,
    )
    return resultat.stdout if resultat.returncode == 0 and resultat.stdout else None


def decrire_scenes(video: Path, vision: dict[str, Any] | None, duree: float,
                   on_progress: Callable[[float], None] | None = None) -> list[dict[str, Any]]:
    """[{debut, fin, description}] pour chaque scène, ou [] sans modèle de vision disponible."""
    from pipeline.translate import _decharger_ollama, ollama_disponible

    modele = get_runtime().modele_vision
    if not modele:
        logger.info("Pas de modèle de vision (processeur seul) : scènes non décrites")
        return []
    if not ollama_disponible(modele):
        return []
    plans = [n / vision["ips"] for n in vision.get("plans", [])] if vision else []
    if not plans:  # vision.json ancien : un repère toutes les 8 s
        plans = [8.0 * k for k in range(1, int(duree // 8) + 1)]
    fenetres = _fenetres(plans, duree)
    scenes: list[dict[str, Any]] = []
    try:
        with httpx.Client(base_url=get_settings().ollama_url, timeout=180) as client:
            for n, (debut, fin) in enumerate(fenetres):
                image = _image(video, (debut + fin) / 2)
                if image is None:
                    continue
                reponse = client.post("/api/chat", json={
                    "model": modele, "stream": False, "keep_alive": "2m", "options": {"temperature": 0.2},
                    "messages": [{"role": "user", "content": PROMPT_SCENE, "images": [base64.b64encode(image).decode()]}],
                })
                reponse.raise_for_status()
                texte = reponse.json()["message"]["content"].strip().replace("\n", " ")[:400]
                if texte:
                    scenes.append({"debut": round(debut, 2), "fin": round(fin, 2), "description": texte})
                if on_progress:
                    on_progress((n + 1) / len(fenetres))
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("Description des scènes interrompue (%s) : %d scène(s) décrite(s)", exc, len(scenes))
    finally:
        _decharger_ollama(modele)
    logger.info("%d scène(s) décrite(s) par %s", len(scenes), modele)
    return scenes


def scene_a(scenes: list[dict[str, Any]], instant: float) -> str | None:
    """Description de la scène en cours à `instant`."""
    for scene in scenes or []:
        if scene["debut"] <= instant < scene["fin"] + 0.01:
            return scene["description"]
    return None


def rediger_fiche(segments: list[dict[str, Any]], locuteurs: dict[str, dict[str, Any]],
                  scenes: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    """Fiche de la vidéo rédigée par le modèle de langue, ou None s'il est indisponible."""
    from pipeline.emotions import decrire
    from pipeline.translate import LANGUAGE_NAMES, _decharger_ollama, ollama_disponible

    modele = get_runtime().ollama_model
    if modele == "nllb" or not ollama_disponible(modele):
        return None
    lignes = []
    for seg in segments:
        nom = (locuteurs.get(seg.get("locuteur") or "", {}) or {}).get("nom", "?")
        emotion = decrire(seg.get("emotion"))
        lignes.append(f"[{seg['debut']:.0f}s] {nom}{f' ({emotion})' if emotion else ''} : {seg['texte']}")
    transcription = "\n".join(lignes)[:TRANSCRIPTION_MAX]
    images = "\n".join(f"[{s['debut']:.0f}-{s['fin']:.0f}s] {s['description']}" for s in scenes)[:3000]
    prompt = "\n".join([
        f"Langue de la vidéo : {LANGUAGE_NAMES.get(source, source)}.",
        "Tu prépares le doublage d'une vidéo. Lis la transcription (avec le locuteur et l'émotion de chaque "
        "réplique) et la description des images, puis rédige en français une fiche de travail pour l'adaptateur :",
        "- genre (ex. : sketch comique, dispute de couple, tutoriel, interview…) ;",
        "- resume : ce qui se passe, en 2 ou 3 phrases ;",
        "- registre : niveau de langue (argot, familier, courant, soutenu) et particularités (verlan, jeunes, accent…) ;",
        "- ton : ambiance générale (humour, tension, émotion…) ;",
        "- personnages : pour chaque locuteur (garder exactement son nom, ex. « Locuteur 2 »), qui il semble être, "
        "son âge apparent, son caractère et sa relation avec les autres.",
        "N'invente rien : appuie-toi seulement sur la transcription et les images ; si une information "
        "(âge, métier, relation…) n'est pas claire, écris « incertain ».",
        "", "DESCRIPTION DES IMAGES :", images or "(non disponible)", "", "TRANSCRIPTION :", transcription,
    ])
    try:
        with httpx.Client(base_url=get_settings().ollama_url, timeout=300) as client:
            reponse = client.post("/api/chat", json={
                "model": modele, "stream": False, "keep_alive": "2m", "format": SCHEMA_FICHE,
                "options": {"temperature": 0.3}, "messages": [{"role": "user", "content": prompt}],
            })
            reponse.raise_for_status()
            fiche = json.loads(reponse.json()["message"]["content"])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("Fiche de la vidéo impossible (%s)", exc)
        return None
    finally:
        _decharger_ollama(modele)
    logger.info("Fiche de la vidéo : %s", fiche.get("genre"))
    return fiche


def formater_fiche(fiche: dict[str, Any] | None) -> str:
    """Fiche en texte, pour la traduction."""
    if not fiche:
        return ""
    lignes = [f"Genre : {fiche.get('genre', '')}", f"Résumé : {fiche.get('resume', '')}",
              f"Registre : {fiche.get('registre', '')}", f"Ton : {fiche.get('ton', '')}"]
    personnages = fiche.get("personnages") or []
    if personnages:
        lignes.append("Personnages :")
        lignes += [f"- {p.get('locuteur', '?')} : {p.get('description', '')}" for p in personnages]
    return "\n".join(lignes)
