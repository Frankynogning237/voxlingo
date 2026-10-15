"""Extraction de l'audio de la vidéo avec FFmpeg (WAV mono 16 kHz)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("voxlingo.extract")


class FFmpegError(RuntimeError):
    """Erreur lisible renvoyée quand FFmpeg ou ffprobe échoue."""


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegError(
            f"{tool} est introuvable. Installez FFmpeg (winget install Gyan.FFmpeg) "
            "puis redémarrez le terminal."
        )
    return path


def probe_duration(video: Path) -> float:
    """Durée de la vidéo en secondes, lue par ffprobe."""
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError("Impossible de lire la vidéo : le fichier est peut-être corrompu.")
    return float(json.loads(result.stdout)["format"]["duration"])


def has_audio(video: Path) -> bool:
    """Indique si la vidéo contient au moins une piste audio."""
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def make_thumbnail(video: Path, output: Path, duree_s: float) -> Path:
    """Capture une image de la vidéo (au tiers, au plus à 3 s) pour les cartes de projet."""
    instant = min(3.0, duree_s / 3)
    result = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{instant:.2f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning("Miniature impossible : %s", result.stderr.strip())
    return output


def compute_peaks(audio: Path, nombre: int = 160) -> list[float]:
    """Amplitudes réelles de l'audio (0 à 1), réparties en `nombre` barres pour la forme d'onde."""
    import wave

    import numpy as np

    with wave.open(str(audio), "rb") as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if samples.size == 0:
        return [0.0] * nombre
    blocs = np.array_split(np.abs(samples.astype(np.float32)), nombre)
    peaks = np.array([b.max() if b.size else 0.0 for b in blocs])
    maximum = peaks.max() or 1.0
    return [round(float(p / maximum), 3) for p in peaks]


def extract_audio(video: Path, output: Path) -> Path:
    """Extrait la piste audio en WAV mono 16 kHz, format attendu par Whisper."""
    if not has_audio(video):
        raise FFmpegError("Cette vidéo ne contient pas de piste audio.")
    result = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("FFmpeg : %s", result.stderr.strip())
        raise FFmpegError("L'extraction de l'audio a échoué (voir les journaux du serveur).")
    return output
