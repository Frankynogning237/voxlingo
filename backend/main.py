"""Serveur local FastAPI de VoxLingo (http://localhost:8000)."""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jobs
from config import get_runtime, get_settings
from pipeline.extract import make_filmstrip
from pipeline.segments import est_trop_long, marquer_fin_max

_LOG_DIR = get_settings().data_dir / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "voxlingo.log", encoding="utf-8"),
    ],
)
# Les requêtes HTTP vers Hugging Face et Ollama encombrent les journaux
for _bruyant in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(_bruyant).setLevel(logging.WARNING)
logger = logging.getLogger("voxlingo")

app = FastAPI(title="VoxLingo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _first_line(cmd: list[str]) -> str | None:
    """Exécute une commande et renvoie la première ligne de sa sortie."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else None


@lru_cache
def tool_versions() -> dict[str, str | None]:
    """Versions des outils externes utilisés par le pipeline."""
    try:
        import torch

        torch_version: str | None = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "ffmpeg": _first_line(["ffmpeg", "-version"]),
        "ollama": _first_line(["ollama", "--version"]),
    }


@app.on_event("startup")
def startup() -> None:
    runtime = get_runtime()
    logger.info("Démarrage : device=%s, mode=%s", runtime.device, runtime.mode)
    # Une tâche "en cours" au démarrage a été interrompue par l'arrêt du serveur
    for status in jobs.list_jobs():
        if status["etat"] in ("en_attente", "en_cours"):
            jobs.update_status(
                status["id"], etat="erreur",
                erreur="Traitement interrompu par l'arrêt du serveur.",
            )


# Formats lus par FFmpeg. Aucune limite de taille ni de durée : seul l'espace disque compte.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}
CHUNK_BYTES = 8 * 1024**2
ESPACE_MINIMAL = 2 * 1024**3  # garder au moins 2 Go libres pour l'audio et les fichiers générés


def _espace_libre() -> int:
    return shutil.disk_usage(get_settings().projects_dir).free


@app.post("/jobs", status_code=201)
async def create_job(
    video: UploadFile = File(...),
    langue_cible: str = Form(...),
    conserver_fond: bool = Form(True),
    synchro_labiale: bool = Form(False),
    sous_titres: bool = Form(True),
    volume_fond: int = Form(100, ge=0, le=100),
    ajuster_trop_longs: bool = Form(True),
) -> dict:
    """Reçoit une vidéo et la langue cible, enregistre le projet et lance le pipeline."""
    nom = Path(video.filename or "video").name
    ext = Path(nom).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, "Format non pris en charge : utilisez MP4, MOV, MKV, WEBM, AVI, M4V, WMV, FLV, TS ou MPG."
        )
    if langue_cible not in get_settings().target_languages:
        raise HTTPException(400, f"Langue cible non prise en charge : {langue_cible}.")

    job_id = jobs.new_job_id()
    folder = jobs.project_dir(job_id)
    folder.mkdir(parents=True)
    destination = folder / f"input{ext}"

    # Écriture par morceaux pour ne jamais charger la vidéo entière en mémoire
    taille = 0
    try:
        with destination.open("wb") as out:
            while chunk := await video.read(CHUNK_BYTES):
                taille += len(chunk)
                # Vérification régulière (tous les ~256 Mo) de l'espace disque restant
                if taille % (32 * CHUNK_BYTES) < len(chunk) and _espace_libre() < ESPACE_MINIMAL:
                    raise HTTPException(
                        507,
                        "Espace disque insuffisant pour cette vidéo : libérez de la place "
                        f"(au moins {ESPACE_MINIMAL // 1024**3} Go doivent rester libres).",
                    )
                out.write(chunk)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    logger.info("Vidéo reçue : %s (%.1f Mo) → tâche %s", nom, taille / 1024**2, job_id)
    options = {
        "conserver_fond": conserver_fond,
        "synchro_labiale": synchro_labiale,
        "sous_titres": sous_titres,
        "volume_fond": volume_fond,
        "ajuster_trop_longs": ajuster_trop_longs,
    }
    status = jobs.create_job(job_id, nom, destination.name, langue_cible, options)
    jobs.start_job(job_id)
    return status


@app.get("/jobs")
def list_jobs() -> list[dict]:
    """Historique des tâches."""
    return jobs.list_jobs()


def _get_status_or_404(job_id: str) -> dict:
    status = jobs.read_status(job_id) if job_id.isalnum() else None
    if status is None:
        raise HTTPException(404, "Projet introuvable.")
    return status


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Statut d'une tâche : état, étape courante, progression, erreurs."""
    return _get_status_or_404(job_id)


@app.get("/jobs/{job_id}/segments")
def get_segments(job_id: str) -> dict:
    """Transcription et traductions, segment par segment."""
    _get_status_or_404(job_id)
    status = _get_status_or_404(job_id)
    if not (jobs.project_dir(job_id) / "segments.json").is_file():
        raise HTTPException(404, "La transcription n'est pas encore disponible.")
    data = jobs.read_segments(job_id)
    if data["segments"] and "fin_max" not in data["segments"][0]:
        marquer_fin_max(data["segments"], status.get("duree_s"))  # projets créés avant l'ajustement
    return data


class TraductionModifiee(BaseModel):
    id: int
    traduction: str


class EditionTraductions(BaseModel):
    segments: list[TraductionModifiee]


def _require_editable(job_id: str) -> dict:
    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Un traitement est en cours : attendez sa fin pour modifier la traduction.")
    if not status.get("moteur_traduction"):
        raise HTTPException(409, "La traduction n'a pas encore été générée.")
    return status


@app.put("/jobs/{job_id}/segments")
def save_translations(job_id: str, edition: EditionTraductions) -> dict:
    """Enregistre les traductions modifiées (brouillon) et recalcule les alertes « Trop long »."""
    status = _require_editable(job_id)
    data = jobs.read_segments(job_id)
    marquer_fin_max(data["segments"], status.get("duree_s"))
    par_id = {e.id: e.traduction.strip() for e in edition.segments}
    for seg in data["segments"]:
        if seg["id"] in par_id:
            if not par_id[seg["id"]]:
                raise HTTPException(400, f"La traduction du segment {seg['id'] + 1} est vide.")
            seg["traduction"] = par_id[seg["id"]]
            seg["trop_long"] = est_trop_long(seg, seg["traduction"], status["langue_source"], status["langue_cible"])
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False, brouillon_le=jobs._now())
    return data


@app.post("/jobs/{job_id}/reinitialiser")
def reset_translations(job_id: str) -> dict:
    """Revient aux traductions proposées par l'IA."""
    status = _require_editable(job_id)
    data = jobs.read_segments(job_id)
    marquer_fin_max(data["segments"], status.get("duree_s"))
    for seg in data["segments"]:
        seg["traduction"] = seg["traduction_ia"]
        seg["trop_long"] = est_trop_long(seg, seg["traduction"], status["langue_source"], status["langue_cible"])
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False)
    return data


@app.post("/jobs/{job_id}/traduire")
def retranslate(job_id: str) -> dict:
    """Relance la traduction par l'IA (écrase les modifications)."""
    _require_editable(job_id)
    status = jobs.update_status(job_id, etat="en_attente")
    jobs.start_translation(job_id)
    return status


class DemandeAjustement(BaseModel):
    ids: list[int] | None = None  # None = toutes les phrases trop longues


@app.post("/jobs/{job_id}/ajuster")
def adjust_translations(job_id: str, demande: DemandeAjustement) -> dict:
    """Ajuste automatiquement les phrases trop longues : silence suivant, raccourcissement, découpage."""
    _require_editable(job_id)
    status = jobs.update_status(job_id, etat="en_attente")
    jobs.start_adjustment(job_id, set(demande.ids) if demande.ids is not None else None)
    return status


@app.post("/jobs/{job_id}/valider")
def validate_translations(job_id: str, edition: EditionTraductions) -> dict:
    """Enregistre puis valide les traductions : la génération des voix pourra démarrer (phase 4)."""
    save_translations(job_id, edition)
    return jobs.update_status(job_id, traduction_validee=True, etat="termine", valide_le=jobs._now())


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """Vidéo d'origine, pour l'aperçu dans l'interface."""
    status = _get_status_or_404(job_id)
    return FileResponse(jobs.project_dir(job_id) / status["fichier"])


@app.get("/jobs/{job_id}/thumbnail")
def get_thumbnail(job_id: str) -> FileResponse:
    """Miniature de la vidéo (générée pendant l'extraction)."""
    _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "thumbnail.jpg"
    if not path.is_file():
        raise HTTPException(404, "Miniature pas encore disponible.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/filmstrip")
def get_filmstrip(job_id: str) -> FileResponse:
    """Bande d'images de la vidéo pour la timeline (créée à la demande pour les anciens projets)."""
    status = _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "filmstrip.jpg"
    if not path.is_file():
        if not status.get("duree_s"):
            raise HTTPException(404, "Bande d'images pas encore disponible.")
        make_filmstrip(jobs.project_dir(job_id) / status["fichier"], path, status["duree_s"])
    if not path.is_file():
        raise HTTPException(500, "Impossible de créer la bande d'images.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/waveform")
def get_waveform(job_id: str) -> list[float]:
    """Amplitudes réelles de l'audio, pour dessiner la forme d'onde."""
    _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "waveform.json"
    if not path.is_file():
        raise HTTPException(404, "Forme d'onde pas encore disponible.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/jobs/{job_id}/annuler")
def cancel_job(job_id: str) -> dict:
    """Demande l'arrêt du traitement en cours."""
    status = _get_status_or_404(job_id)
    if status["etat"] not in ("en_attente", "en_cours"):
        raise HTTPException(409, "Ce traitement n'est pas en cours.")
    return jobs.request_cancel(job_id)


@app.get("/health")
def health() -> dict:
    """État du système : matériel, mode, modèles choisis et outils."""
    runtime = get_runtime()
    vram_libre = 0.0
    if runtime.device == "cuda":
        import torch

        libre, _total = torch.cuda.mem_get_info()
        vram_libre = round(libre / 1024**3, 1)
    return {
        "status": "ok",
        "device": runtime.device,
        "mode": runtime.mode,
        "vram_totale_go": runtime.vram_go,
        "vram_libre_go": vram_libre,
        "modeles": {
            "transcription": runtime.whisper_model,
            "traduction": runtime.ollama_model,
            "synchro_labiale_locale": runtime.lipsync_local,
        },
        "langues_cibles": get_settings().target_languages,
        "nom_utilisateur": get_settings().nom_utilisateur,
        "outils": tool_versions(),
    }
