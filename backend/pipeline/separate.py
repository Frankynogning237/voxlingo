"""Séparation voix / fond sonore : RoFormer si disponible, sinon Demucs (htdemucs, deux pistes).

Produit, dans le dossier du projet :
  voix_isolee.wav   la voix seule (meilleure transcription, meilleur clonage) ;
  fond.wav          musique et ambiance d'origine, remixées sous la voix doublée.

RoFormer (paquet facultatif `audio-separator`, modèle BS-RoFormer) isole la voix nettement plus
proprement que Demucs : moins d'écho de la voix d'origine dans le fond, échantillon de clonage plus
net. S'il n'est pas installé ou échoue, Demucs prend le relais.

Chaque modèle s'exécute dans un processus séparé : un manque de mémoire n'arrête que ce processus.
Les deux découpent l'audio en morceaux, ce qui garde une mémoire bornée quelle que soit la durée.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from config import get_runtime, get_settings
from pipeline.extract import FFmpegError, _require

logger = logging.getLogger("voxlingo.separate")

MODELE = "htdemucs"
# BS-RoFormer (viperx) : l'un des meilleurs modèles ouverts pour isoler la voix
MODELE_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class SeparationError(RuntimeError):
    """Erreur lisible de la séparation voix / fond."""


def separer(video: Path, dossier: Path, on_progress: Callable[[float], None] | None = None) -> tuple[Path, Path]:
    """Sépare la voix et le fond sonore de la vidéo. Renvoie (voix_isolee.wav, fond.wav)."""
    voix, fond = dossier / "voix_isolee.wav", dossier / "fond.wav"
    if voix.is_file() and fond.is_file():
        return voix, fond

    # Audio d'origine en qualité studio (44,1 kHz stéréo), attendu par RoFormer et Demucs
    source = dossier / "audio_hq.wav"
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(source)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("Impossible d'extraire l'audio pour la séparation voix / fond.")

    try:
        if not _roformer(source, dossier, voix, fond, on_progress):
            _demucs(source, dossier, voix, fond, on_progress)
    finally:
        source.unlink(missing_ok=True)
    return voix, fond


def _executer(commande: list[str], on_progress: Callable[[float], None] | None,
              env: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """Lance un modèle dans un processus séparé et relaie sa barre de progression « 42%| … »."""
    import os

    processus = subprocess.Popen(
        commande, cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, **(env or {})},
    )
    journal: list[str] = []
    try:
        for ligne in processus.stdout:
            journal.append(ligne)
            for pourcentage in re.findall(r"(\d{1,3})%\|", ligne):
                if on_progress:
                    on_progress(min(int(pourcentage), 100) / 100)
        processus.wait()
    except BaseException:
        processus.kill()  # annulation : on arrête le modèle
        raise
    return processus.returncode, journal


def roformer_disponible() -> bool:
    import importlib.util

    return importlib.util.find_spec("audio_separator") is not None


def _roformer(source: Path, dossier: Path, voix: Path, fond: Path,
              on_progress: Callable[[float], None] | None) -> bool:
    """Séparation RoFormer. Renvoie False (Demucs prendra le relais) si elle est impossible."""
    if not roformer_disponible():
        return False
    sortie = dossier / "roformer"
    modeles = get_settings().models_dir / "audio-separator"
    logger.info("RoFormer %s (%s)", MODELE_ROFORMER, get_runtime().device)
    code, journal = _executer(
        [sys.executable, "-X", "utf8", "-m", "pipeline.roformer_worker", str(source), str(sortie), str(modeles),
         MODELE_ROFORMER],
        on_progress,
    )
    if code != 0 or not (sortie / "vocals.wav").is_file() or not (sortie / "no_vocals.wav").is_file():
        logger.warning("RoFormer indisponible, passage à Demucs : %s", "".join(journal[-10:]))
        shutil.rmtree(sortie, ignore_errors=True)
        return False
    shutil.move(str(sortie / "vocals.wav"), voix)
    shutil.move(str(sortie / "no_vocals.wav"), fond)
    shutil.rmtree(sortie, ignore_errors=True)
    return True


def _demucs(source: Path, dossier: Path, voix: Path, fond: Path,
            on_progress: Callable[[float], None] | None) -> None:
    sortie = dossier / "demucs"
    commande = [
        sys.executable, "-X", "utf8", "-m", "pipeline.demucs_worker", "--two-stems", "vocals", "-n", MODELE,
        "-d", get_runtime().device, "-o", str(sortie), "--filename", "{stem}.{ext}", str(source),
    ]
    logger.info("Demucs %s (%s)", MODELE, get_runtime().device)
    code, journal = _executer(commande, on_progress, {"TORCH_HOME": str(get_settings().models_dir / "torch")})
    if code != 0:
        logger.error("Demucs : %s", "".join(journal[-20:]))
        raise SeparationError(
            "La séparation voix / fond a échoué (manque de mémoire ?). Fermez les autres applications, "
            "ou désactivez « Conserver le fond sonore » pour ce projet."
        )

    dossier_stems = sortie / MODELE
    shutil.move(str(dossier_stems / "vocals.wav"), voix)
    shutil.move(str(dossier_stems / "no_vocals.wav"), fond)
    shutil.rmtree(sortie, ignore_errors=True)
