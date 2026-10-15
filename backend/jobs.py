"""Gestion des tâches de doublage : création, exécution en arrière-plan, statut en JSON.

Chaque tâche correspond à un dossier `data/projects/<id>/` contenant la vidéo
d'origine, les fichiers intermédiaires et `status.json`, qui est la source de
vérité lue par l'API (GET /jobs/{id}).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import get_settings

logger = logging.getLogger("voxlingo.jobs")

JobState = Literal["en_attente", "en_cours", "a_valider", "termine", "erreur", "annule"]
StepState = Literal["attente", "en_cours", "termine", "erreur", "a_venir"]

# Étapes affichées dans l'interface, dans l'ordre du pipeline.
# `disponible` passe à True au fil des phases de développement.
STEPS: list[dict[str, Any]] = [
    {"cle": "extraction", "libelle": "Extraction audio", "disponible": True},
    {"cle": "separation", "libelle": "Séparation voix / fond", "disponible": False},
    {"cle": "transcription", "libelle": "Transcription", "disponible": True},
    {"cle": "locuteurs", "libelle": "Identification des locuteurs", "disponible": False},
    {"cle": "traduction", "libelle": "Traduction", "disponible": True},
    {"cle": "voix", "libelle": "Génération des voix", "disponible": False},
    {"cle": "synchro_labiale", "libelle": "Synchronisation labiale", "disponible": False},
    {"cle": "export", "libelle": "Export", "disponible": False},
]

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_dir(job_id: str) -> Path:
    return get_settings().projects_dir / job_id


def _status_path(job_id: str) -> Path:
    return project_dir(job_id) / "status.json"


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def create_job(
    job_id: str, nom_fichier: str, fichier: str, langue_cible: str, options: dict[str, Any]
) -> dict[str, Any]:
    """Crée le status.json initial d'une tâche dont la vidéo est déjà enregistrée."""
    status: dict[str, Any] = {
        "id": job_id,
        "nom_fichier": nom_fichier,
        "fichier": fichier,
        "langue_cible": langue_cible,
        "options": options,
        "langue_source": None,
        "probabilite_langue": None,
        "duree_s": None,
        "etat": "en_attente",
        "etape": None,
        "progression": 0,
        "progression_etape": 0,
        "erreur": None,
        "annulation_demandee": False,
        "moteur_traduction": None,
        "traduction_validee": False,
        "dernier_ajustement": None,
        "debut_traitement": None,
        "fin_traitement": None,
        "cree_le": _now(),
        "maj_le": _now(),
        "etapes": [
            {
                "cle": s["cle"],
                "libelle": s["libelle"],
                "etat": "attente" if s["disponible"] else "a_venir",
                "duree_s": None,
            }
            for s in STEPS
        ],
    }
    write_status(job_id, status)
    return status


def read_status(job_id: str) -> dict[str, Any] | None:
    path = _status_path(job_id)
    if not path.is_file():
        return None
    with _lock:
        return json.loads(path.read_text(encoding="utf-8"))


def write_status(job_id: str, status: dict[str, Any]) -> None:
    """Écrit status.json de façon atomique (fichier temporaire puis remplacement)."""
    status["maj_le"] = _now()
    path = _status_path(job_id)
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def update_status(job_id: str, **changes: Any) -> dict[str, Any]:
    status = read_status(job_id)
    if status is None:
        raise KeyError(job_id)
    status.update(changes)
    write_status(job_id, status)
    return status


def update_step(job_id: str, cle: str, **changes: Any) -> None:
    status = read_status(job_id)
    if status is None:
        raise KeyError(job_id)
    for step in status["etapes"]:
        if step["cle"] == cle:
            step.update(changes)
    write_status(job_id, status)


def _segments_path(job_id: str) -> Path:
    return project_dir(job_id) / "segments.json"


def read_segments(job_id: str) -> dict[str, Any]:
    """Transcription et traductions : {langue, duree_s, segments: [{id, debut, fin, texte, traduction…}]}."""
    path = _segments_path(job_id)
    with _lock:
        return json.loads(path.read_text(encoding="utf-8"))


def write_segments(job_id: str, data: dict[str, Any]) -> None:
    path = _segments_path(job_id)
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def list_jobs() -> list[dict[str, Any]]:
    """Historique des tâches, de la plus récente à la plus ancienne."""
    jobs = []
    for path in get_settings().projects_dir.glob("*/status.json"):
        status = read_status(path.parent.name)
        if status:
            jobs.append(status)
    return sorted(jobs, key=lambda s: s["cree_le"], reverse=True)


def start_job(job_id: str) -> None:
    """Lance le pipeline de la tâche dans un thread en arrière-plan."""
    from pipeline.runner import run_pipeline

    thread = threading.Thread(target=run_pipeline, args=(job_id,), name=f"job-{job_id}", daemon=True)
    thread.start()
    logger.info("Tâche %s lancée", job_id)


def start_translation(job_id: str) -> None:
    """Relance uniquement l'étape de traduction dans un thread en arrière-plan."""
    from pipeline.runner import run_translation

    threading.Thread(target=run_translation, args=(job_id,), name=f"trad-{job_id}", daemon=True).start()


def start_adjustment(job_id: str, ids: set[int] | None) -> None:
    """Lance l'ajustement des phrases trop longues dans un thread en arrière-plan."""
    from pipeline.runner import run_adjustment

    threading.Thread(target=run_adjustment, args=(job_id, ids), name=f"ajust-{job_id}", daemon=True).start()


class JobCancelled(RuntimeError):
    """Levée dans le pipeline quand l'utilisateur a demandé l'annulation."""


def request_cancel(job_id: str) -> dict[str, Any]:
    """Demande l'annulation : le pipeline s'arrête au prochain point de contrôle."""
    return update_status(job_id, annulation_demandee=True)


def check_cancelled(job_id: str) -> None:
    status = read_status(job_id)
    if status and status.get("annulation_demandee"):
        raise JobCancelled("Traitement annulé par l'utilisateur.")
