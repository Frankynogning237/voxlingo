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


def couper_video(video: Path, sortie: Path, debut_s: float, fin_s: float) -> Path:
    """Garde seulement le passage [debut_s, fin_s] de la vidéo (option « Raccourcir la vidéo »).

    La vidéo est réencodée (H.264 + AAC, MP4) pour une coupe exacte à l'image près, sans
    décalage entre l'image et le son ; seules les premières pistes vidéo et audio sont gardées.
    """
    result = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{debut_s:.3f}", "-i", str(video),
         "-t", f"{fin_s - debut_s:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(sortie)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not sortie.is_file():
        raise FFmpegError(f"Impossible de raccourcir la vidéo : {result.stderr.strip()[-300:]}")
    return sortie


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


def make_filmstrip(video: Path, output: Path, duree_s: float, images: int = 24) -> Path:
    """Bande d'images régulièrement espacées (une seule image JPEG) pour la timeline de l'éditeur.

    Chaque image est prise par un accès direct à son instant (-ss avant -i) : le temps de
    calcul ne dépend pas de la durée de la vidéo, même pour un film de plusieurs heures.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    ffmpeg = _require("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="voxlingo-bande-") as dossier:
        def capturer(i: int) -> None:
            instant = (i + 0.5) * duree_s / images
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-ss", f"{instant:.2f}", "-i", str(video),
                 "-frames:v", "1", "-vf", "scale=160:90:force_original_aspect_ratio=increase,crop=160:90",
                 "-q:v", "5", str(Path(dossier) / f"image_{i:03d}.jpg")],
                capture_output=True, text=True,
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(capturer, range(images)))
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-start_number", "0", "-i", str(Path(dossier) / "image_%03d.jpg"),
             "-vf", f"tile={images}x1", "-frames:v", "1", "-q:v", "5", str(output)],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        logger.warning("Bande d'images impossible : %s", result.stderr.strip())
    return output


def compute_peaks(audio: Path, nombre: int = 1200) -> list[float]:
    """Amplitudes réelles de l'audio (0 à 1), réparties en `nombre` barres pour la forme d'onde.

    L'audio est lu bloc par bloc : la mémoire utilisée reste faible quelle que soit la durée.
    """
    import wave

    import numpy as np

    with wave.open(str(audio), "rb") as wav:
        total = wav.getnframes()
        if total == 0:
            return [0.0] * nombre
        peaks = []
        for i in range(nombre):
            taille = (total * (i + 1)) // nombre - (total * i) // nombre
            bloc = np.frombuffer(wav.readframes(taille), dtype=np.int16)
            peaks.append(float(np.abs(bloc.astype(np.int32)).max()) if bloc.size else 0.0)
    maximum = max(peaks) or 1.0
    return [round(p / maximum, 3) for p in peaks]


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
