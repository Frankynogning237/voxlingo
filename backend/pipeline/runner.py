"""Orchestration du pipeline : enchaîne les étapes, met à jour le statut et la progression."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

import jobs
from pipeline.extract import compute_peaks, extract_audio, make_thumbnail, probe_duration
from pipeline.transcribe import transcribe

logger = logging.getLogger("voxlingo.runner")

# Poids relatif de chaque étape disponible dans la progression globale (en %).
WEIGHTS: dict[str, int] = {
    "extraction": 5,
    "transcription": 95,
}


def _progress_reporter(job_id: str, cle: str) -> Callable[[float], None]:
    """Renvoie une fonction qui convertit l'avancement d'une étape (0..1) en progression globale."""
    done_before = 0
    for key, weight in WEIGHTS.items():
        if key == cle:
            break
        done_before += weight
    weight = WEIGHTS[cle]
    last = {"etape": -1}

    def report(fraction: float) -> None:
        jobs.check_cancelled(job_id)  # point de contrôle de l'annulation
        etape = int(fraction * 100)
        if etape != last["etape"]:  # évite d'écrire status.json pour rien
            last["etape"] = etape
            jobs.update_status(
                job_id, progression=int(done_before + weight * fraction), progression_etape=etape
            )

    return report


def _run_step(job_id: str, cle: str, action: Callable[[Callable[[float], None]], None]) -> None:
    """Exécute une étape en journalisant sa durée et en mettant à jour son état."""
    libelle = next(s["libelle"] for s in jobs.STEPS if s["cle"] == cle)
    logger.info("[%s] %s : début", job_id, libelle)
    jobs.update_status(job_id, etape=cle, progression_etape=0)
    jobs.update_step(job_id, cle, etat="en_cours")
    report = _progress_reporter(job_id, cle)
    start = time.perf_counter()
    try:
        action(report)
    except Exception:
        jobs.update_step(job_id, cle, etat="erreur", duree_s=round(time.perf_counter() - start, 1))
        raise
    duree = round(time.perf_counter() - start, 1)
    report(1.0)
    jobs.update_step(job_id, cle, etat="termine", duree_s=duree)
    logger.info("[%s] %s : terminé en %.1f s", job_id, libelle, duree)


def run_pipeline(job_id: str) -> None:
    """Point d'entrée du thread de traitement d'une tâche."""
    status = jobs.read_status(job_id)
    if status is None:
        logger.error("Tâche %s introuvable", job_id)
        return

    folder = jobs.project_dir(job_id)
    video = folder / status["fichier"]
    audio = folder / "audio.wav"
    jobs.update_status(
        job_id, etat="en_cours", progression=0, erreur=None, debut_traitement=jobs._now()
    )

    try:
        def do_extract(report: Callable[[float], None]) -> None:
            duree = probe_duration(video)
            jobs.update_status(job_id, duree_s=round(duree, 2))
            make_thumbnail(video, folder / "thumbnail.jpg", duree)
            report(0.3)
            extract_audio(video, audio)
            (folder / "waveform.json").write_text(json.dumps(compute_peaks(audio)), encoding="utf-8")

        def do_transcribe(report: Callable[[float], None]) -> None:
            result = transcribe(audio, on_progress=report)
            (folder / "segments.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            jobs.update_status(
                job_id,
                langue_source=result.langue,
                probabilite_langue=result.probabilite,
            )

        _run_step(job_id, "extraction", do_extract)
        _run_step(job_id, "transcription", do_transcribe)

        jobs.update_status(
            job_id, etat="termine", etape=None, progression=100, fin_traitement=jobs._now()
        )
        logger.info("[%s] Pipeline terminé", job_id)
    except jobs.JobCancelled as exc:
        logger.info("[%s] Annulé", job_id)
        jobs.update_status(job_id, etat="annule", etape=None, erreur=str(exc), fin_traitement=jobs._now())
    except Exception as exc:  # toute erreur est remontée à l'interface
        logger.exception("[%s] Échec du pipeline", job_id)
        message = str(exc) if str(exc) else exc.__class__.__name__
        jobs.update_status(job_id, etat="erreur", erreur=message, fin_traitement=jobs._now())
