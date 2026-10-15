"""Configuration centralisée du backend VoxLingo.

Les valeurs par défaut peuvent être surchargées dans le fichier `.env`
à la racine du projet (voir `.env.example`).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger("voxlingo.config")

Device = Literal["cuda", "cpu"]
Mode = Literal["gpu_local", "gpu_limite", "cpu"]


def detect_device() -> tuple[Device, float]:
    """Renvoie le device disponible et la VRAM totale en Go (0 en CPU)."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch n'est pas installé : passage en mode CPU.")
        return "cpu", 0.0

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return "cuda", round(vram, 1)
    return "cpu", 0.0


def detect_mode(device: Device, vram_go: float) -> Mode:
    """Choisit le mode de fonctionnement à partir du matériel."""
    if device == "cuda":
        return "gpu_local" if vram_go >= 8 else "gpu_limite"
    return "cpu"


class Settings(BaseSettings):
    """Paramètres de l'application, surchargeables par `.env`."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Matériel : "auto" détecte la carte NVIDIA, sinon on force cuda ou cpu.
    force_device: Literal["auto", "cuda", "cpu"] = "auto"

    # Dossiers
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"

    # Modèles (None = choix automatique selon le mode)
    whisper_model: str | None = None
    ollama_model: str | None = None
    ollama_url: str = "http://localhost:11434"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Langues cibles proposées dans l'interface (supportées par XTTS v2)
    target_languages: list[str] = [
        "fr", "en", "es", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    ]

    # Jeton Hugging Face (phase 6), uniquement dans .env
    hf_token: str | None = None

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"


class Runtime:
    """Matériel détecté et modèles choisis, calculés une seule fois."""

    def __init__(self, settings: Settings) -> None:
        detected, vram = detect_device()
        if settings.force_device == "auto":
            self.device: Device = detected
        elif settings.force_device == "cuda" and detected != "cuda":
            logger.warning("CUDA demandé mais aucune carte NVIDIA détectée : mode CPU.")
            self.device = "cpu"
        else:
            self.device = settings.force_device
        self.vram_go = vram if self.device == "cuda" else 0.0
        self.mode: Mode = detect_mode(self.device, self.vram_go)

        # Modèles adaptés au mode (surchargeables dans .env)
        self.whisper_model = settings.whisper_model or (
            "large-v3-turbo" if self.device == "cuda" else "small"
        )
        self.whisper_compute_type = "float16" if self.device == "cuda" else "int8"
        self.ollama_model = settings.ollama_model or (
            "qwen2.5:7b" if self.mode == "gpu_local" else "qwen2.5:3b"
        )
        self.lipsync_local = self.mode == "gpu_local"

        if self.device == "cpu":
            logger.warning(
                "Aucune carte NVIDIA utilisée : traitement sur CPU, plus lent. "
                "La synchronisation labiale passera par Kaggle."
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache
def get_runtime() -> Runtime:
    return Runtime(get_settings())
