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
StepState = Literal["attente", "en_cours", "termine", "erreur", "a_venir", "ignore", "kaggle"]

# Étapes affichées dans l'interface, dans l'ordre du pipeline.
# `disponible` passe à True au fil des phases de développement.
STEPS: list[dict[str, Any]] = [
    {"cle": "extraction", "libelle": "Extraction audio", "disponible": True},
    {"cle": "separation", "libelle": "Séparation voix / fond", "disponible": True},
    {"cle": "transcription", "libelle": "Transcription", "disponible": True},
    {"cle": "locuteurs", "libelle": "Identification des locuteurs", "disponible": True},
    {"cle": "analyse", "libelle": "Analyse de la scène (émotions, contexte)", "disponible": True},
    {"cle": "traduction", "libelle": "Traduction", "disponible": True},
    {"cle": "voix", "libelle": "Génération des voix", "disponible": True},
    {"cle": "synchro_labiale", "libelle": "Synchronisation labiale", "disponible": True},
    {"cle": "export", "libelle": "Export", "disponible": True},
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
        "fichier_sortie": None,
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


def doublage_pret(status: dict[str, Any]) -> bool:
    """Vrai si la vidéo doublée est exportée et qu'il ne restait que la synchronisation labiale."""
    etapes = {s["cle"]: s["etat"] for s in status.get("etapes", [])}
    sortie = status.get("fichier_sortie")
    return (status.get("etape") == "synchro_labiale" and etapes.get("export") == "termine"
            and bool(sortie) and (project_dir(status["id"]) / sortie).is_file())


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


def strategie(status: dict[str, Any]) -> str:
    """Stratégie face aux phrases trop longues : « reformuler » (par défaut) ou « accelerer »."""
    from pipeline.segments import STRATEGIE_PAR_DEFAUT, STRATEGIES

    choix = (status.get("options") or {}).get("strategie_duree")
    return choix if choix in STRATEGIES else STRATEGIE_PAR_DEFAUT


LOCUTEUR_PAR_DEFAUT = "SPEAKER_00"


def locuteur(seg: dict[str, Any]) -> str:
    """Locuteur d'un segment (projets antérieurs à l'identification : un seul locuteur)."""
    return seg.get("locuteur") or LOCUTEUR_PAR_DEFAUT


def locuteurs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fiche des locuteurs d'un projet : {code: {nom, couleur, voix, f0_hz}}."""
    fiche = data.get("locuteurs") or {}
    for seg in data.get("segments", []):
        fiche.setdefault(locuteur(seg), {"nom": f"Locuteur {len(fiche) + 1}", "couleur": "#a855f7"})
    for infos in fiche.values():
        infos.setdefault("voix", "clone")
    return fiche


def list_jobs() -> list[dict[str, Any]]:
    """Historique des tâches, de la plus récente à la plus ancienne."""
    jobs = []
    for path in get_settings().projects_dir.glob("*/status.json"):
        status = read_status(path.parent.name)
        if status:
            jobs.append(status)
    return sorted(jobs, key=lambda s: s["cree_le"], reverse=True)


def delete_job(job_id: str) -> None:
    """Supprime définitivement le projet et tous ses fichiers."""
    import shutil

    shutil.rmtree(project_dir(job_id))


def duplicate_job(job_id: str, langue_cible: str | None = None) -> dict[str, Any]:
    """Crée un nouveau projet avec la même vidéo et les mêmes réglages (dans une autre langue cible
    si `langue_cible` est donnée), puis lance le traitement."""
    import shutil

    source = read_status(job_id)
    if source is None:
        raise KeyError(job_id)
    nouveau = new_job_id()
    dossier = project_dir(nouveau)
    dossier.mkdir(parents=True)
    shutil.copy2(project_dir(job_id) / source["fichier"], dossier / source["fichier"])
    options = dict(source.get("options") or {})
    if source.get("decoupe_appliquee"):
        options.pop("decoupe", None)  # la vidéo copiée est déjà raccourcie
    status = create_job(nouveau, source["nom_fichier"], source["fichier"], langue_cible or source["langue_cible"],
                        options)
    start_job(nouveau)
    return status


def start_job(job_id: str) -> None:
    """Lance le pipeline de la tâche dans un thread en arrière-plan."""
    from pipeline.runner import run_pipeline

    thread = threading.Thread(target=run_pipeline, args=(job_id,), name=f"job-{job_id}", daemon=True)
    thread.start()
    logger.info("Tâche %s lancée", job_id)


def start_translation(job_id: str, ids: set[int] | None = None) -> None:
    """Relance l'étape de traduction (toutes les phrases ou seulement `ids`) en arrière-plan."""
    from pipeline.runner import run_translation

    threading.Thread(target=run_translation, args=(job_id, ids), name=f"trad-{job_id}", daemon=True).start()


def start_adjustment(job_id: str, ids: set[int] | None) -> None:
    """Lance l'ajustement des phrases trop longues dans un thread en arrière-plan."""
    from pipeline.runner import run_adjustment

    threading.Thread(target=run_adjustment, args=(job_id, ids), name=f"ajust-{job_id}", daemon=True).start()


def start_voices(job_id: str) -> None:
    """Lance la génération des voix et l'export dans un thread en arrière-plan."""
    from pipeline.runner import run_voices

    threading.Thread(target=run_voices, args=(job_id,), name=f"voix-{job_id}", daemon=True).start()


def start_analysis(job_id: str) -> None:
    """Lance l'analyse de la scène (émotions, scènes, fiche) dans un thread en arrière-plan."""
    from pipeline.runner import run_analysis

    threading.Thread(target=run_analysis, args=(job_id,), name=f"analyse-{job_id}", daemon=True).start()


def start_remix(job_id: str) -> None:
    """Refait le mixage du son (voix existantes) dans un thread en arrière-plan."""
    from pipeline.runner import run_remix

    threading.Thread(target=run_remix, args=(job_id,), name=f"mixage-{job_id}", daemon=True).start()


def start_lipsync(job_id: str) -> None:
    """Lance la synchronisation labiale par le moteur dans un thread en arrière-plan."""
    from pipeline.runner import run_lipsync

    threading.Thread(target=run_lipsync, args=(job_id,), name=f"synchro-{job_id}", daemon=True).start()


def start_habillage(job_id: str) -> None:
    """Refait la vidéo habillée (sous-titres incrustés, flou, logo…) dans un thread en arrière-plan."""
    from pipeline.runner import habiller_projet

    threading.Thread(target=habiller_projet, args=(job_id,), name=f"habillage-{job_id}", daemon=True).start()


def start_speakers(job_id: str) -> None:
    """Lance l'identification des locuteurs dans un thread en arrière-plan."""
    from pipeline.runner import run_speakers

    threading.Thread(target=run_speakers, args=(job_id,), name=f"locuteurs-{job_id}", daemon=True).start()


class JobCancelled(RuntimeError):
    """Levée dans le pipeline quand l'utilisateur a demandé l'annulation."""


def request_cancel(job_id: str) -> dict[str, Any]:
    """Demande l'annulation : le pipeline s'arrête au prochain point de contrôle."""
    return update_status(job_id, annulation_demandee=True)


def check_cancelled(job_id: str) -> None:
    status = read_status(job_id)
    if status and status.get("annulation_demandee"):
        raise JobCancelled("Traitement annulé par l'utilisateur.")
