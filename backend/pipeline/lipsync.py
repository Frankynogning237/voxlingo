"""Synchronisation labiale dans le moteur, avec MuseTalk 1.5 (Tencent Music, licence MIT).

Quand le moteur tourne sur GPU (Kaggle, Colab), les lèvres de la vidéo doublée sont recalées sur
la nouvelle voix sans passer par le carnet manuel (kaggle/lipsync_kaggle.ipynb, qui reste la
solution sans GPU). MuseTalk exige Python 3.10 et PyTorch 2.0 : il vit dans son propre
environnement (réglages MUSETALK_PYTHON et MUSETALK_DIR), installé en arrière-plan par le carnet.

Son script charge toute la vidéo en mémoire : on ne lui confie que des morceaux courts (≤ 15 s),
pris là où une voix doublée parle. Le reste de l'image est gardé tel quel, puis tout est recollé
et la piste doublée remise par-dessus. Mémoire bornée, quelle que soit la durée de la vidéo.
Un morceau sans visage exploitable (MuseTalk échoue) garde simplement l'image d'origine.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import get_settings
from pipeline.extract import FFmpegError, _require

logger = logging.getLogger("voxlingo.lipsync")

MORCEAU_MAX_S = 15.0  # durée maximale confiée à MuseTalk d'un coup (mémoire)
MARGE_S = 0.2  # avant et après chaque phrase
ECART_FUSION_S = 0.6  # deux phrases plus proches que ça sont traitées ensemble
SILENCE_MAX_S = 20 * 60  # MuseTalk muet et sans nouveau passage depuis 20 min : figé, on l'arrête
SONDAGE_S = 2.0  # intervalle de suivi de MuseTalk
CADENCE = 25  # images/s des morceaux (la vidéo finale est remise à la cadence d'origine)
# Réencodage commun à tous les morceaux, pour pouvoir les recoller sans réencoder une seconde fois
ENCODAGE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-r", str(CADENCE), "-an"]


class LipsyncError(RuntimeError):
    """Erreur lisible de la synchronisation labiale."""


def musetalk_disponible() -> bool:
    """Vrai si l'environnement de MuseTalk est installé et ses poids téléchargés (fichier « .pret »)."""
    settings = get_settings()
    if not settings.musetalk_python or not settings.musetalk_dir:
        return False
    return Path(settings.musetalk_python).is_file() and (Path(settings.musetalk_dir) / ".pret").is_file()


def fenetres(segments: list[dict[str, Any]], duree: float, max_s: float = MORCEAU_MAX_S,
             marge: float = MARGE_S, ecart: float = ECART_FUSION_S) -> list[tuple[float, float]]:
    """Intervalles (début, fin) où une voix doublée parle, fusionnés puis découpés en morceaux ≤ max_s."""
    intervalles = sorted(
        (max(0.0, s["debut"] - marge), min(duree, max(s["fin"], s.get("fin_max") or 0.0) + marge))
        for s in segments if (s.get("traduction") or "").strip()
    )
    fusion: list[list[float]] = []
    for debut, fin in intervalles:
        if fin <= debut:
            continue
        if fusion and debut - fusion[-1][1] < ecart:
            fusion[-1][1] = max(fusion[-1][1], fin)
        else:
            fusion.append([debut, fin])
    morceaux: list[tuple[float, float]] = []
    for debut, fin in fusion:
        n = max(1, int(-(-(fin - debut) // max_s)))  # nombre de morceaux, arrondi au-dessus
        pas = (fin - debut) / n
        morceaux += [(round(debut + i * pas, 3), round(debut + (i + 1) * pas, 3)) for i in range(n)]
    return morceaux


def plan_de_montage(morceaux: list[tuple[float, float]], duree: float) -> list[tuple[float, float, int | None]]:
    """Toute la durée en tronçons consécutifs : (début, fin, n° du morceau synchronisé ou None)."""
    plan: list[tuple[float, float, int | None]] = []
    curseur = 0.0
    for i, (debut, fin) in enumerate(morceaux):
        if debut - curseur > 0.001:  # entre deux passages synchronisés : l'image d'origine
            plan.append((curseur, debut, None))
        plan.append((max(curseur, debut), fin, i))
        curseur = fin
    if duree - curseur > 0.001:
        plan.append((curseur, duree, None))
    return plan


def _ffmpeg(*arguments: str) -> None:
    resultat = subprocess.run([_require("ffmpeg"), "-y", "-v", "error", *arguments], capture_output=True, text=True)
    if resultat.returncode != 0:
        raise FFmpegError(f"FFmpeg (synchronisation labiale) : {resultat.stderr.strip()[-300:]}")


def images(debut: float, fin: float) -> int:
    """Nombre exact d'images d'un tronçon : les durées sont comptées en images pour ne pas dériver."""
    return max(1, round(fin * CADENCE) - round(debut * CADENCE))


def sur_la_grille(t: float) -> float:
    """Instant arrondi à l'image (1/25 s) la plus proche."""
    return round(t * CADENCE) / CADENCE


class _Sortie:
    """Sortie de MuseTalk lue au fil de l'eau : dernières lignes, dernier pourcentage, dernier signe de vie."""

    def __init__(self) -> None:
        self.lignes: deque[str] = deque(maxlen=40)
        self.pourcent = 0
        self.activite = time.monotonic()


def _suivre_sortie(processus: subprocess.Popen) -> _Sortie:
    """Lit la sortie dans un fil à part : la progression avance même quand MuseTalk n'écrit rien."""
    sortie = _Sortie()

    def lire() -> None:
        for ligne in processus.stdout:  # les barres tqdm (« \r ») comptent comme des lignes
            sortie.lignes.append(ligne)
            sortie.activite = time.monotonic()
            if barre := re.search(r"(\d{1,3})%\|", ligne):
                sortie.pourcent = min(int(barre.group(1)), 100)

    threading.Thread(target=lire, name="musetalk-sortie", daemon=True).start()
    return sortie


def _attendre_musetalk(processus: subprocess.Popen, sortie: _Sortie, resultats: Path, total: int,
                       rapporter: Callable[[float], None]) -> None:
    """Suit MuseTalk jusqu'à sa fin : passages finis, plus la barre du passage en cours. Sans aucun signe de
    vie (ni sortie ni nouveau passage) pendant SILENCE_MAX_S, il est considéré figé."""
    faits_avant, avance = 0, 0.1
    while processus.poll() is None:
        faits = len(list(resultats.glob("synchro_*.mp4"))) if resultats.is_dir() else 0
        if faits != faits_avant:
            faits_avant, sortie.activite, sortie.pourcent = faits, time.monotonic(), 0
        # Un passage compte trois barres (repères du visage, calcul, recollage) : la sienne n'en est qu'un tiers
        en_cours = min(sortie.pourcent / 100 / 3, 0.9) if faits < total else 0
        avance = max(avance, 0.1 + 0.75 * (faits + en_cours) / total)
        rapporter(avance)  # peut lever JobCancelled ; tient aussi l'état du projet à jour
        if time.monotonic() - sortie.activite > SILENCE_MAX_S:
            raise LipsyncError(f"MuseTalk ne répond plus depuis {SILENCE_MAX_S // 60} min "
                               f"({faits}/{total} passage(s) synchronisé(s)) : il a été arrêté.")
        time.sleep(SONDAGE_S)


def _extrait_video(video: Path, debut: float, fin: float, taille: str, sortie: Path) -> None:
    _ffmpeg("-ss", f"{debut:.3f}", "-i", str(video), "-vf", f"{taille},tpad=stop_mode=clone:stop_duration=1",
            "-frames:v", str(images(debut, fin)), *ENCODAGE, str(sortie))


def synchroniser(video: Path, piste: Path, segments: list[dict[str, Any]], duree: float, dossier: Path,
                 on_progress: Callable[[float], None] | None = None) -> Path:
    """Vidéo doublée aux lèvres synchronisées (MP4, 25 images/s, piste `piste`), dans `dossier`."""
    settings = get_settings()
    travail = dossier / "synchro"
    if travail.exists():
        import shutil

        shutil.rmtree(travail)
    (travail / "morceaux").mkdir(parents=True)
    rapporter = on_progress or (lambda _f: None)
    morceaux = [(sur_la_grille(d), sur_la_grille(f)) for d, f in fenetres(segments, duree)]
    morceaux = [(d, f) for d, f in morceaux if f > d]
    if not morceaux:
        raise LipsyncError("Aucune phrase doublée : rien à synchroniser.")
    taille = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # dimensions paires, exigées par H.264

    # 1. Morceaux de vidéo (image seule) et de voix doublée confiés à MuseTalk
    taches = []
    for i, (debut, fin) in enumerate(morceaux):
        extrait, voix = travail / f"source_{i:04d}.mp4", travail / f"voix_{i:04d}.wav"
        _extrait_video(video, debut, fin, taille, extrait)
        _ffmpeg("-ss", f"{debut:.3f}", "-i", str(piste), "-t", f"{fin - debut:.3f}", "-ac", "1", "-ar", "16000",
                str(voix))
        taches.append(f'tache_{i:04d}:\n  video_path: "{extrait}"\n  audio_path: "{voix}"\n'
                      f'  result_name: "synchro_{i:04d}.mp4"\n')
    configuration = travail / "taches.yaml"
    configuration.write_text("".join(taches), encoding="utf-8")
    rapporter(0.1)

    # 2. MuseTalk, en une seule fois (modèles chargés une fois), dans son propre environnement
    resultats = travail / "resultats"
    commande = [
        settings.musetalk_python, "-m", "scripts.inference", "--inference_config", str(configuration),
        "--result_dir", str(resultats), "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json", "--version", "v15", "--use_float16",
    ]
    logger.info("MuseTalk : %d morceau(x) à synchroniser", len(morceaux))
    import os

    # Dans un carnet Jupyter, MPLBACKEND désigne l'affichage du carnet, inconnu de cet environnement :
    # matplotlib (importé par mmpose) refuserait de démarrer
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    processus = subprocess.Popen(commande, cwd=settings.musetalk_dir, stdout=subprocess.PIPE, env=env,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    journal = _suivre_sortie(processus)
    try:
        _attendre_musetalk(processus, journal, resultats / "v15", len(morceaux), rapporter)
    except BaseException:
        processus.kill()  # annulation ou MuseTalk figé : on l'arrête
        raise
    synchronises = {i for i in range(len(morceaux)) if (resultats / "v15" / f"synchro_{i:04d}.mp4").is_file()}
    if not synchronises:
        logger.error("MuseTalk : %s", "".join(list(journal.lignes)[-20:]))
        raise LipsyncError("MuseTalk n'a synchronisé aucun passage (visages introuvables ou manque de mémoire).")
    logger.info("MuseTalk : %d/%d morceau(x) synchronisé(s)", len(synchronises), len(morceaux))

    # 3. Recollage : passages synchronisés + image d'origine ailleurs, puis la piste doublée
    liste = []
    for n, (debut, fin, i) in enumerate(plan_de_montage(morceaux, sur_la_grille(duree))):
        troncon = travail / "morceaux" / f"{n:05d}.mp4"
        if i is not None and i in synchronises:
            # Même nombre d'images que l'original : un résultat un peu court est complété par sa dernière image
            _ffmpeg("-i", str(resultats / "v15" / f"synchro_{i:04d}.mp4"),
                    "-vf", f"{taille},tpad=stop_mode=clone:stop_duration=1",
                    "-frames:v", str(images(debut, fin)), *ENCODAGE, str(troncon))
        else:
            _extrait_video(video, debut, fin, taille, troncon)
        liste.append(f"file '{troncon.as_posix()}'\n")
    (travail / "liste.txt").write_text("".join(liste), encoding="utf-8")
    image = travail / "image.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(travail / "liste.txt"), "-c", "copy", str(image))
    rapporter(0.95)
    sortie = dossier / "synchro_moteur.mp4"
    _ffmpeg("-i", str(image), "-i", str(piste), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-shortest", str(sortie))
    import shutil

    shutil.rmtree(travail, ignore_errors=True)
    return sortie
