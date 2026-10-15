"""Orchestration du pipeline : enchaîne les étapes, met à jour le statut et la progression.

Le pipeline s'arrête après la traduction (statut « a_valider ») : l'utilisateur
relit et corrige les traductions avant la génération des voix.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

import jobs
from pipeline.extract import compute_peaks, extract_audio, make_filmstrip, make_thumbnail, probe_duration
from pipeline.transcribe import transcribe
from pipeline.adjust import ajuster_segments
from pipeline.segments import est_trop_long, marquer_fin_max
from pipeline.translate import translate_segments

logger = logging.getLogger("voxlingo.runner")

# Poids relatif de chaque étape disponible dans la progression globale (en %).
WEIGHTS: dict[str, int] = {
    "extraction": 5,
    "transcription": 55,
    "traduction": 40,
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
    jobs.update_step(job_id, cle, etat="en_cours", duree_s=None)
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


def _guarded(job_id: str, body: Callable[[], None]) -> None:
    """Exécute `body` en transformant annulation et erreurs en statut lisible."""
    jobs.update_status(
        job_id, etat="en_cours", erreur=None, annulation_demandee=False, debut_traitement=jobs._now()
    )
    try:
        body()
        jobs.update_status(
            job_id, etat="a_valider", etape=None, fin_traitement=jobs._now()
        )
        logger.info("[%s] Traduction prête, en attente de validation", job_id)
    except jobs.JobCancelled as exc:
        logger.info("[%s] Annulé", job_id)
        jobs.update_status(job_id, etat="annule", etape=None, erreur=str(exc), fin_traitement=jobs._now())
    except Exception as exc:  # toute erreur est remontée à l'interface
        logger.exception("[%s] Échec du pipeline", job_id)
        message = str(exc) if str(exc) else exc.__class__.__name__
        jobs.update_status(job_id, etat="erreur", erreur=message, fin_traitement=jobs._now())


# Part de l'étape « Traduction » consacrée à l'ajustement des phrases trop longues
PART_AJUSTEMENT = 0.2


def _sous_progression(report: Callable[[float], None], debut: float, fin: float) -> Callable[[float], None]:
    return lambda fraction: report(debut + (fin - debut) * fraction)


def _do_translate(job_id: str, report: Callable[[float], None]) -> None:
    """Traduit tous les segments, puis ajuste les phrases trop longues si l'option est active."""
    status = jobs.read_status(job_id)
    source, cible = status["langue_source"], status["langue_cible"]
    ajuster = status.get("options", {}).get("ajuster_trop_longs", True)
    data = jobs.read_segments(job_id)
    data["duree_s"] = status.get("duree_s") or data.get("duree_s")
    marquer_fin_max(data["segments"], data["duree_s"])

    fin_traduction = 1 - PART_AJUSTEMENT if ajuster else 1.0
    traductions, moteur = translate_segments(
        data["segments"], source, cible, on_progress=_sous_progression(report, 0, fin_traduction)
    )
    for seg, texte in zip(data["segments"], traductions, strict=True):
        seg["traduction_ia"] = texte
        seg["traduction"] = texte
        seg["trop_long"] = est_trop_long(seg, texte, source, cible)
    data["langue_cible"] = cible
    data["moteur_traduction"] = moteur
    jobs.write_segments(job_id, data)

    bilan = None
    if ajuster and any(seg["trop_long"] for seg in data["segments"]):
        bilan = ajuster_segments(data, source, cible, on_progress=_sous_progression(report, fin_traduction, 1))
        jobs.write_segments(job_id, data)
    jobs.update_status(job_id, moteur_traduction=moteur, traduction_validee=False, dernier_ajustement=bilan)


def _do_adjust(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    """Ajuste les phrases trop longues après les modifications de l'utilisateur."""
    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    data["duree_s"] = status.get("duree_s") or data.get("duree_s")
    bilan = ajuster_segments(data, status["langue_source"], status["langue_cible"], ids=ids, on_progress=report)
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False, dernier_ajustement=bilan)


def run_pipeline(job_id: str) -> None:
    """Point d'entrée du thread de traitement d'une nouvelle tâche."""
    status = jobs.read_status(job_id)
    if status is None:
        logger.error("Tâche %s introuvable", job_id)
        return

    folder = jobs.project_dir(job_id)
    video = folder / status["fichier"]
    audio = folder / "audio.wav"
    jobs.update_status(job_id, progression=0)

    def do_extract(report: Callable[[float], None]) -> None:
        duree = probe_duration(video)
        jobs.update_status(job_id, duree_s=round(duree, 2))
        make_thumbnail(video, folder / "thumbnail.jpg", duree)
        make_filmstrip(video, folder / "filmstrip.jpg", duree)
        report(0.3)
        extract_audio(video, audio)
        (folder / "waveform.json").write_text(json.dumps(compute_peaks(audio)), encoding="utf-8")

    def do_transcribe(report: Callable[[float], None]) -> None:
        result = transcribe(audio, on_progress=report)
        jobs.write_segments(job_id, result.to_dict())
        jobs.update_status(job_id, langue_source=result.langue, probabilite_langue=result.probabilite)

    def body() -> None:
        _run_step(job_id, "extraction", do_extract)
        _run_step(job_id, "transcription", do_transcribe)
        _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report))

    _guarded(job_id, body)


def run_adjustment(job_id: str, ids: set[int] | None = None) -> None:
    """Ajuste les phrases trop longues (bouton « Ajuster automatiquement » de l'éditeur)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_adjust(job_id, ids, report)))


def run_translation(job_id: str) -> None:
    """Relance uniquement la traduction (bouton « Traduction assistée par IA »)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report)))
