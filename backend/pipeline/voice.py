"""Clonage de voix et synthèse de chaque segment traduit avec XTTS v2.

Licence : le modèle XTTS v2 est distribué sous Coqui Public Model License (CPML),
qui n'autorise que l'usage personnel non commercial (voir README.md). Son
acceptation est déclarée ici, conformément à CONSIGNE-lancement-doublage-video.md.

Les voix générées sont gardées en cache (voix/brut_<id>.wav + manifeste) : après
une correction, seules les phrases dont le texte a changé sont régénérées.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import get_runtime, get_settings
from pipeline.extract import _require
from pipeline.segments import vitesse_xtts

logger = logging.getLogger("voxlingo.voice")

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML acceptée pour un usage personnel
os.environ.setdefault("TTS_HOME", str(get_settings().models_dir / "tts"))

# Windows : torchcodec (lecture audio de PyTorch ≥ 2.9) a besoin des DLL FFmpeg partagées,
# installées dans l'environnement conda (conda install -c conda-forge ffmpeg).
if sys.platform == "win32":
    _dll_ffmpeg = Path(sys.prefix) / "Library" / "bin"
    if _dll_ffmpeg.is_dir():
        os.add_dll_directory(str(_dll_ffmpeg))

TAUX = 24000  # fréquence de sortie de XTTS v2
REFERENCE_MIN_S = 6.0
REFERENCE_MAX_S = 15.0
SEUIL_SILENCE = 0.01  # amplitude sous laquelle le début et la fin sont considérés silencieux


class VoiceError(RuntimeError):
    """Erreur lisible de la génération des voix."""


# ───────────────────────── Échantillon de référence ─────────────────────────

def extraire_reference(video: Path, segments: list[dict[str, Any]], sortie: Path) -> Path:
    """Assemble 6 à 15 s de voix d'origine (les phrases les plus longues) en un échantillon propre."""
    import soundfile as sf

    candidats = sorted(
        (s for s in segments if 1.5 <= s["fin"] - s["debut"] <= 12),
        key=lambda s: s["fin"] - s["debut"],
        reverse=True,
    ) or sorted(segments, key=lambda s: s["fin"] - s["debut"], reverse=True)
    choisis, total = [], 0.0
    for seg in candidats:
        if total >= REFERENCE_MIN_S + 2 or total + (seg["fin"] - seg["debut"]) > REFERENCE_MAX_S:
            continue
        choisis.append(seg)
        total += seg["fin"] - seg["debut"]
    if not choisis:
        raise VoiceError("Aucune parole exploitable pour cloner la voix.")

    morceaux = []
    for seg in sorted(choisis, key=lambda s: s["debut"]):
        resultat = subprocess.run(
            [_require("ffmpeg"), "-v", "error", "-ss", f"{seg['debut']:.2f}", "-to", f"{seg['fin']:.2f}",
             "-i", str(video), "-vn", "-ac", "1", "-ar", str(TAUX), "-af", "highpass=f=70",
             "-f", "s16le", "-"],
            capture_output=True,
        )
        if resultat.returncode == 0 and resultat.stdout:
            morceaux.append(np.frombuffer(resultat.stdout, dtype=np.int16))
            morceaux.append(np.zeros(int(0.25 * TAUX), dtype=np.int16))  # courte pause entre extraits
    if not morceaux:
        raise VoiceError("Impossible d'extraire l'échantillon de voix d'origine.")
    sf.write(sortie, np.concatenate(morceaux), TAUX, subtype="PCM_16")
    logger.info("Échantillon de référence : %.1f s (%d extraits)", total, len(choisis))
    return sortie


# ───────────────────────── Synthèse ─────────────────────────

def _rogner_silences(wav: np.ndarray) -> np.ndarray:
    """Retire les silences ajoutés par XTTS au début et à la fin (garde 50 ms)."""
    actif = np.flatnonzero(np.abs(wav) > SEUIL_SILENCE)
    if actif.size == 0:
        return wav
    marge = int(0.05 * TAUX)
    return wav[max(0, actif[0] - marge): min(len(wav), actif[-1] + marge)]


class Manifeste:
    """Texte et langue de chaque voix déjà générée, pour ne régénérer que ce qui a changé."""

    def __init__(self, dossier: Path) -> None:
        self.chemin = dossier / "manifeste.json"
        self.donnees: dict[str, dict[str, str]] = (
            json.loads(self.chemin.read_text(encoding="utf-8")) if self.chemin.is_file() else {}
        )

    @staticmethod
    def _cle(seg: dict[str, Any], langue: str) -> dict[str, Any]:
        # Le temps disponible fixe la vitesse de lecture : s'il change, la voix est refaite
        disponible = round(seg.get("fin_max", seg["fin"]) - seg["debut"], 2)
        return {"texte": seg["traduction"], "langue": langue, "disponible": disponible}

    def a_jour(self, seg: dict[str, Any], langue: str, fichier: Path) -> bool:
        entree = self.donnees.get(str(seg["id"]), {})
        return fichier.is_file() and {k: entree.get(k) for k in ("texte", "langue", "disponible")} == self._cle(seg, langue)

    def noter(self, seg: dict[str, Any], langue: str, vitesse: float | None = None) -> None:
        self.donnees[str(seg["id"])] = {**self._cle(seg, langue), "vitesse": vitesse}
        self.chemin.write_text(json.dumps(self.donnees, ensure_ascii=False, indent=1), encoding="utf-8")


def fichier_brut(dossier: Path, seg_id: int) -> Path:
    return dossier / f"brut_{seg_id:05d}.wav"


MEMOIRE_CONSEILLEE_GO = 3.0  # XTTS v2 sur processeur


def _memoire_libre_go() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return float("nan")


def generer_voix(
    segments: list[dict[str, Any]],
    langue: str,
    reference: Path,
    dossier: Path,
    on_progress: Callable[[float], None] | None = None,
) -> int:
    """Synthétise chaque segment traduit avec la voix clonée. Renvoie le nombre de phrases générées.

    La synthèse tourne dans un processus séparé (pipeline/voice_worker.py) : un manque de
    mémoire n'arrête que ce processus, et devient un message d'erreur lisible.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    manifeste = Manifeste(dossier)
    a_generer = [s for s in segments if not manifeste.a_jour(s, langue, fichier_brut(dossier, s["id"]))]
    logger.info("Voix : %d phrase(s) à générer, %d déjà à jour", len(a_generer), len(segments) - len(a_generer))
    if not a_generer:
        return 0

    libre = _memoire_libre_go()
    if libre < MEMOIRE_CONSEILLEE_GO:
        logger.warning("Seulement %.1f Go de mémoire libre (%.0f Go conseillés pour XTTS) : "
                       "fermez les autres applications si la génération échoue.", libre, MEMOIRE_CONSEILLEE_GO)

    tache = dossier / "tache.json"
    tache.write_text(json.dumps({
        "segments": [
            {
                "id": s["id"],
                "traduction": s["traduction"],
                "vitesse": vitesse_xtts(s, s["traduction"], langue),
                "disponible": max(s.get("fin_max", s["fin"]) - s["debut"], 0.1),
            }
            for s in a_generer
        ],
        "langue": langue,
        "reference": str(reference),
        "dossier": str(dossier),
    }, ensure_ascii=False), encoding="utf-8")
    par_id = {s["id"]: s for s in a_generer}

    logger.info("Chargement de XTTS v2 dans un processus séparé (%s)", get_runtime().device)
    processus = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "pipeline.voice_worker", str(tache)],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    faits = 0
    try:
        for ligne in processus.stdout:
            ligne = ligne.strip()
            if ligne == "PRET":
                logger.info("XTTS chargé, génération de %d phrase(s)", len(a_generer))
            elif ligne.startswith("FAIT "):
                _, ident, vitesse = ligne.split()
                seg = par_id[int(ident)]
                manifeste.noter(seg, langue, float(vitesse))
                faits += 1
                logger.info("Voix %d/%d (vitesse ×%s) : %s", faits, len(a_generer), vitesse, seg["traduction"][:60])
                if on_progress:
                    on_progress(faits / len(a_generer))  # peut lever JobCancelled
        processus.wait()
    except BaseException:
        processus.kill()  # annulation ou erreur : on arrête la synthèse
        raise
    finally:
        tache.unlink(missing_ok=True)

    if processus.returncode != 0:
        erreurs = processus.stderr.read().strip()
        logger.error("Processus XTTS terminé avec le code %s : %s", processus.returncode, erreurs[-1500:])
        message = next((l[7:] for l in reversed(erreurs.splitlines()) if l.startswith("ERREUR ")), None)
        if message is None or "memory" in message.lower() or "alloc" in message.lower():
            raise VoiceError(
                f"La génération des voix s'est arrêtée brutalement après {faits} phrase(s), très probablement "
                f"par manque de mémoire ({libre:.1f} Go libres au départ, {MEMOIRE_CONSEILLEE_GO:.0f} Go conseillés). "
                "Fermez les autres applications (navigateur, Word…) puis cliquez sur « Valider et générer les voix » : "
                "les phrases déjà générées sont conservées."
            )
        raise VoiceError(f"La génération des voix a échoué : {message}")
    return faits
