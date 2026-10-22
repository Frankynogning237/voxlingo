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
    # Modèle de vision qui décrit les scènes (Ollama) ; vide = automatique (GPU seulement), « aucun » = désactivé
    modele_vision: str | None = None
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Langues cibles proposées dans l'interface (supportées par XTTS v2)
    target_languages: list[str] = [
        "fr", "en", "es", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    ]

    # Prénom affiché dans l'interface ("Bonjour …")
    nom_utilisateur: str = ""

    # Jeton Hugging Face (phase 6), uniquement dans .env
    hf_token: str | None = None
    # Relecture du dialogue traduit (cohérence, accords, naturel) : plus fine, ≈ 1,5 fois plus longue
    relecture_traduction: bool = True
    # Égalisation du timbre des voix clonées vers la voix d'origine (pipeline/timbre_eq.py) : mesurée de
    # 100 Hz à 6,3 kHz et gardée seulement si la voix ressemble au moins autant au locuteur
    egalisation_timbre: bool = True
    # Sauvegarde automatique sur Hugging Face (moteur sur Kaggle/Colab) : jeton avec droit
    # d'écriture et dépôt dataset privé (par défaut « votre-compte/voxlingo-sauvegarde »)
    hf_token_ecriture: str | None = None
    hf_depot_sauvegarde: str | None = None
    # Python de l'environnement séparé de Chatterbox (voix plus expressive, PyTorch 2.6) ; vide = XTTS seul
    chatterbox_python: str | None = None
    # Synchronisation labiale dans le moteur (MuseTalk 1.5, environnement séparé installé par le carnet)
    musetalk_python: str | None = None
    musetalk_dir: str | None = None

    # Accès à distance (moteur sur Kaggle/Colab, interface sur Vercel).
    # Clé d'accès : si elle est définie, chaque requête doit la fournir.
    cle_acces: str | None = None
    # Sites autorisés à appeler le moteur (CORS)
    cors_origines: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_regex: str | None = r"https://[a-z0-9-]+\.vercel\.app"

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
        # Large V3 Turbo partout : sur processeur, il transcrit ~1,7 fois plus lentement que Small, mais
        # sans ses contresens (« me filer à ton plan » au lieu de « me fier à ton clan »), et tout le
        # doublage part de cette transcription
        self.whisper_model = settings.whisper_model or "large-v3-turbo"
        self.whisper_compute_type = "float16" if self.device == "cuda" else "int8"
        self.ollama_model = settings.ollama_model or (
            "qwen2.5:7b" if self.mode == "gpu_local" else "qwen2.5:3b"
        )
        self.lipsync_local = self.mode == "gpu_local"
        # Description des scènes : trop lente sur un simple processeur, réservée au GPU par défaut
        self.modele_vision = settings.modele_vision or ("qwen2.5vl:7b" if self.device == "cuda" else None)
        if (self.modele_vision or "").lower() in ("aucun", "none", "non"):  # désactivé explicitement
            self.modele_vision = None

        if self.device == "cpu":
            logger.warning(
                "Aucune carte NVIDIA utilisée : traitement sur CPU, plus lent. "
                "La synchronisation labiale passera par Kaggle."
            )


FICHIER_PARAMETRES = ROOT_DIR / "data" / "parametres.json"

# Réglages modifiables depuis la page Paramètres, avec leur valeur par défaut
PARAMETRES_DEFAUT: dict[str, object] = {
    "force_device": "auto",
    "whisper_model": "auto",
    "ollama_model": "auto",
    "moteur_voix": "auto",  # auto : Chatterbox s'il est installé, sinon XTTS v2
    "strategie_duree": "reformuler",
    "sous_titres": True,
    "conserver_fond": True,
    "volume_fond": 100,
}


def lire_parametres() -> dict[str, object]:
    """Réglages enregistrés depuis l'interface (data/parametres.json), complétés par les défauts."""
    import json

    valeurs = dict(PARAMETRES_DEFAUT)
    if FICHIER_PARAMETRES.is_file():
        try:
            enregistres = json.loads(FICHIER_PARAMETRES.read_text(encoding="utf-8"))
            valeurs.update({k: v for k, v in enregistres.items() if k in PARAMETRES_DEFAUT})
        except ValueError:
            logger.warning("parametres.json illisible : valeurs par défaut utilisées.")
    return valeurs


def ecrire_parametres(valeurs: dict[str, object]) -> None:
    import json

    FICHIER_PARAMETRES.parent.mkdir(parents=True, exist_ok=True)
    FICHIER_PARAMETRES.write_text(json.dumps(valeurs, ensure_ascii=False, indent=2), encoding="utf-8")
    get_settings.cache_clear()
    get_runtime.cache_clear()


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Les choix faits dans la page Paramètres priment sur .env
    parametres = lire_parametres()
    if parametres["force_device"] != "auto" or settings.force_device == "auto":
        settings.force_device = parametres["force_device"]  # type: ignore[assignment]
    if parametres["whisper_model"] != "auto":
        settings.whisper_model = str(parametres["whisper_model"])
    if parametres["ollama_model"] != "auto":
        settings.ollama_model = str(parametres["ollama_model"])
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache
def get_runtime() -> Runtime:
    return Runtime(get_settings())
