"""Serveur local FastAPI de VoxLingo (http://localhost:8000)."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        "outils": tool_versions(),
    }
