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
    "extraction": 3,
    "separation": 10,
    "transcription": 12,
    "traduction": 20,
    "voix": 45,
    "export": 10,
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
    except jobs.JobCancelled:
        jobs.update_step(job_id, cle, etat="attente", duree_s=None)  # annulée : pas un échec
        raise
    except Exception:
        jobs.update_step(job_id, cle, etat="erreur", duree_s=round(time.perf_counter() - start, 1))
        raise
    duree = round(time.perf_counter() - start, 1)
    report(1.0)
    jobs.update_step(job_id, cle, etat="termine", duree_s=duree)
    logger.info("[%s] %s : terminé en %.1f s", job_id, libelle, duree)


def _reinitialiser_etapes(job_id: str, premiere: str) -> None:
    """Avant une nouvelle exécution : l'étape `premiere` et les suivantes repassent « en attente »."""
    status = jobs.read_status(job_id)
    cles = [s["cle"] for s in status["etapes"]]
    for step in status["etapes"][cles.index(premiere):]:
        if step["etat"] != "a_venir":
            step.update(etat="attente", duree_s=None)
    jobs.write_status(job_id, status)


def _guarded(job_id: str, body: Callable[[], None], etat_final: str = "a_valider") -> None:
    """Exécute `body` en transformant annulation et erreurs en statut lisible."""
    jobs.update_status(
        job_id, etat="en_cours", erreur=None, annulation_demandee=False, debut_traitement=jobs._now()
    )
    try:
        body()
        jobs.update_status(job_id, etat=etat_final, etape=None, fin_traitement=jobs._now())
        logger.info("[%s] Terminé (%s)", job_id, etat_final)
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
        seg["trop_long"] = est_trop_long(seg, texte, source, cible, jobs.strategie(status))
    data["langue_cible"] = cible
    data["moteur_traduction"] = moteur
    jobs.write_segments(job_id, data)

    bilan = None
    if ajuster and any(seg["trop_long"] for seg in data["segments"]):
        bilan = ajuster_segments(data, source, cible, on_progress=_sous_progression(report, fin_traduction, 1),
                                 strategie=jobs.strategie(status))
        jobs.write_segments(job_id, data)
    jobs.update_status(job_id, moteur_traduction=moteur, traduction_validee=False, dernier_ajustement=bilan)


def _do_adjust(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    """Ajuste les phrases trop longues après les modifications de l'utilisateur."""
    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    data["duree_s"] = status.get("duree_s") or data.get("duree_s")
    bilan = ajuster_segments(data, status["langue_source"], status["langue_cible"], ids=ids, on_progress=report,
                             strategie=jobs.strategie(status))
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

    def do_separate(report: Callable[[float], None]) -> None:
        from pipeline.separate import separer

        separer(video, folder, on_progress=report)

    def do_transcribe(report: Callable[[float], None]) -> None:
        voix_isolee = folder / "voix_isolee.wav"
        result = transcribe(voix_isolee if voix_isolee.is_file() else audio, on_progress=report)
        jobs.write_segments(job_id, result.to_dict())
        jobs.update_status(job_id, langue_source=result.langue, probabilite_langue=result.probabilite)

    def body() -> None:
        _run_step(job_id, "extraction", do_extract)
        if (status.get("options") or {}).get("conserver_fond", True):
            _run_step(job_id, "separation", do_separate)
        else:
            jobs.update_step(job_id, "separation", etat="ignore")
        _run_step(job_id, "transcription", do_transcribe)
        _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report))

    _guarded(job_id, body)


def run_adjustment(job_id: str, ids: set[int] | None = None) -> None:
    """Ajuste les phrases trop longues (bouton « Ajuster automatiquement » de l'éditeur)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _reinitialiser_etapes(job_id, "traduction")
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_adjust(job_id, ids, report)))


def run_translation(job_id: str) -> None:
    """Relance uniquement la traduction (bouton « Traduction assistée par IA »)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _reinitialiser_etapes(job_id, "traduction")
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report)))


# ───────────────────────── Phase 4 : voix et export ─────────────────────────

def _do_voices(job_id: str, report: Callable[[float], None]) -> None:
    """Clone la voix d'origine et génère chaque phrase traduite (seules les phrases modifiées sont refaites)."""
    from pipeline.voice import extraire_reference, generer_voix

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    dossier = folder / "voix"
    dossier.mkdir(exist_ok=True)
    data = jobs.read_segments(job_id)
    reference = dossier / "reference.wav"
    if not reference.is_file():
        voix_isolee = folder / "voix_isolee.wav"
        extraire_reference(voix_isolee if voix_isolee.is_file() else folder / status["fichier"],
                           data["segments"], reference)
    report(0.02)
    generees = generer_voix(
        data["segments"], status["langue_cible"], reference, dossier,
        on_progress=_sous_progression(report, 0.02, 1.0),
    )
    jobs.update_status(job_id, voix_generees=generees)


def _do_export(job_id: str, report: Callable[[float], None]) -> None:
    """Cale chaque voix sur son temps, construit la piste doublée et exporte vidéo et sous-titres."""
    from pipeline.align import caler
    from pipeline.assemble import construire_piste, ecrire_srt, exporter_video, mixer_fond
    from pipeline.voice import fichier_brut

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    dossier = folder / "voix"
    data = jobs.read_segments(job_id)
    duree = status.get("duree_s") or data.get("duree_s")
    marquer_fin_max(data["segments"], duree)

    placements = []
    for n, seg in enumerate(data["segments"]):
        cale = dossier / f"cale_{seg['id']:05d}.wav"
        resultat = caler(fichier_brut(dossier, seg["id"]), cale, seg)
        seg["facteur_voix"] = resultat["facteur"]
        seg["duree_doublage"] = resultat["duree"]
        seg["trop_long"] = resultat["trop_long"]  # mesure réelle, remplace l'estimation
        placements.append((seg["debut"], cale))
        report(0.5 * (n + 1) / len(data["segments"]))
    jobs.write_segments(job_id, data)

    construire_piste(placements, duree, folder / "voix_doublee.wav")
    report(0.6)
    options = status.get("options") or {}
    if options.get("conserver_fond", True):
        from pipeline.separate import separer

        _, fond = separer(folder / status["fichier"], folder)  # projets créés avant la phase 5
        mixer_fond(folder / "voix_doublee.wav", fond, int(options.get("volume_fond", 100)), folder / "doublage.wav")
    else:
        (folder / "voix_doublee.wav").replace(folder / "doublage.wav")
    report(0.7)
    exporter_video(folder / status["fichier"], folder / "doublage.wav", folder / "doublage.mp4")
    if status.get("options", {}).get("sous_titres", True):
        ecrire_srt(data["segments"], folder / "sous-titres.srt")
    jobs.update_status(
        job_id,
        fichier_sortie="doublage.mp4",
        segments_trop_longs=sum(1 for s in data["segments"] if s["trop_long"]),
    )


def run_voices(job_id: str) -> None:
    """Génère les voix puis exporte la vidéo doublée (après « Valider et générer les voix »)."""
    jobs.update_status(
        job_id,
        progression=WEIGHTS["extraction"] + WEIGHTS["transcription"] + WEIGHTS["traduction"],
        fichier_sortie=None,
    )
    _reinitialiser_etapes(job_id, "voix")

    def body() -> None:
        _run_step(job_id, "voix", lambda report: _do_voices(job_id, report))
        _run_step(job_id, "export", lambda report: _do_export(job_id, report))

    _guarded(job_id, body, etat_final="termine")
