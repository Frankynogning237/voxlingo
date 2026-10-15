"""Séparation voix / fond sonore avec Demucs (modèle htdemucs, mode deux pistes).

Produit, dans le dossier du projet :
  voix_isolee.wav   la voix seule (meilleure transcription, meilleur clonage) ;
  fond.wav          musique et ambiance d'origine, remixées sous la voix doublée.

Demucs s'exécute dans un processus séparé : comme pour XTTS, un manque de mémoire
n'arrête que ce processus. Il découpe lui-même l'audio en morceaux, ce qui garde une
mémoire bornée quelle que soit la durée de la vidéo.
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


class SeparationError(RuntimeError):
    """Erreur lisible de la séparation voix / fond."""


def separer(video: Path, dossier: Path, on_progress: Callable[[float], None] | None = None) -> tuple[Path, Path]:
    """Sépare la voix et le fond sonore de la vidéo. Renvoie (voix_isolee.wav, fond.wav)."""
    voix, fond = dossier / "voix_isolee.wav", dossier / "fond.wav"
    if voix.is_file() and fond.is_file():
        return voix, fond

    # Audio d'origine en qualité studio (44,1 kHz stéréo), attendu par Demucs
    source = dossier / "audio_hq.wav"
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(source)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("Impossible d'extraire l'audio pour la séparation voix / fond.")

    sortie = dossier / "demucs"
    commande = [
        sys.executable, "-X", "utf8", "-m", "pipeline.demucs_worker", "--two-stems", "vocals", "-n", MODELE,
        "-d", get_runtime().device, "-o", str(sortie), "--filename", "{stem}.{ext}", str(source),
    ]
    env_modeles = {"TORCH_HOME": str(get_settings().models_dir / "torch")}
    logger.info("Demucs %s (%s)", MODELE, get_runtime().device)
    import os

    processus = subprocess.Popen(
        commande, cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", env={**os.environ, **env_modeles},
    )
    journal: list[str] = []
    try:
        # Demucs affiche une barre de progression « 42%| … » : on la relaie
        for ligne in processus.stdout:
            journal.append(ligne)
            for pourcentage in re.findall(r"(\d{1,3})%\|", ligne):
                if on_progress:
                    on_progress(min(int(pourcentage), 100) / 100)
        processus.wait()
    except BaseException:
        processus.kill()  # annulation : on arrête Demucs
        raise
    if processus.returncode != 0:
        logger.error("Demucs : %s", "".join(journal[-20:]))
        raise SeparationError(
            "La séparation voix / fond a échoué (manque de mémoire ?). Fermez les autres applications, "
            "ou désactivez « Conserver le fond sonore » pour ce projet."
        )

    dossier_stems = sortie / MODELE
    shutil.move(str(dossier_stems / "vocals.wav"), voix)
    shutil.move(str(dossier_stems / "no_vocals.wav"), fond)
    shutil.rmtree(sortie, ignore_errors=True)
    source.unlink(missing_ok=True)
    return voix, fond
