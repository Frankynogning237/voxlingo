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

import jobs
from config import get_runtime, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
)
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


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024**3  # 2 Go
CHUNK_BYTES = 8 * 1024**2


@app.post("/jobs", status_code=201)
async def create_job(
    video: UploadFile = File(...),
    langue_cible: str = Form(...),
    conserver_fond: bool = Form(True),
    synchro_labiale: bool = Form(False),
    sous_titres: bool = Form(True),
    volume_fond: int = Form(100, ge=0, le=100),
) -> dict:
    """Reçoit une vidéo et la langue cible, enregistre le projet et lance le pipeline."""
    nom = Path(video.filename or "video").name
    ext = Path(nom).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Format non pris en charge : utilisez MP4, MOV, MKV ou WEBM.")
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
                if taille > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Vidéo trop lourde : 2 Go maximum.")
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
    """Transcription (et plus tard traduction) segment par segment."""
    _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "segments.json"
    if not path.is_file():
        raise HTTPException(404, "La transcription n'est pas encore disponible.")
    return json.loads(path.read_text(encoding="utf-8"))


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
