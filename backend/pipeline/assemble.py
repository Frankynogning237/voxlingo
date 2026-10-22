"""Mixage des voix doublées, export de la vidéo doublée et des sous-titres .srt.

La piste audio est écrite au fil de l'eau : seuls le segment en cours et le
précédent sont en mémoire, quelle que soit la durée de la vidéo.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pipeline.extract import FFmpegError, _require

logger = logging.getLogger("voxlingo.assemble")

TAUX = 24000


def construire_piste(placements: list[tuple[float, Path]], duree_video: float, sortie: Path) -> Path:
    """Place chaque voix (début en secondes, fichier WAV) sur une piste de la durée de la vidéo.

    Les voix sont traitées dans l'ordre. Un tampon garde la dernière voix tant qu'une
    suivante peut encore la chevaucher (les deux sont alors additionnées) ; tout ce qui
    précède le début de la voix courante est écrit sur le disque.
    """
    total = int(round(duree_video * TAUX))
    with sf.SoundFile(sortie, "w", samplerate=TAUX, channels=1, subtype="PCM_16") as piste:
        ecrit = 0
        tampon_debut, tampon = 0, np.zeros(0, dtype=np.float32)

        def ecrire_jusqua(limite: int) -> None:
            """Écrit les échantillons [ecrit, limite) : silence + partie du tampon qui tombe dedans."""
            nonlocal ecrit, tampon_debut, tampon
            limite = min(limite, total)
            if limite <= ecrit:
                return
            bloc = np.zeros(limite - ecrit, dtype=np.float32)
            a, b = max(tampon_debut, ecrit), min(tampon_debut + tampon.size, limite)
            if b > a:
                bloc[a - ecrit: b - ecrit] = tampon[a - tampon_debut: b - tampon_debut]
            piste.write(np.clip(bloc, -1.0, 1.0))
            ecrit = limite
            if tampon_debut + tampon.size <= limite:
                tampon_debut, tampon = limite, np.zeros(0, dtype=np.float32)
            elif tampon_debut < limite:
                tampon, tampon_debut = tampon[limite - tampon_debut:], limite

        for debut_s, fichier in sorted(placements, key=lambda p: p[0]):
            voix, taux = sf.read(fichier, dtype="float32")
            if taux != TAUX:
                raise FFmpegError(f"Fréquence inattendue ({taux} Hz) pour {fichier.name}.")
            debut = int(round(debut_s * TAUX))
            if debut >= total:
                continue
            ecrire_jusqua(debut)
            if tampon.size and debut < tampon_debut + tampon.size:
                # Chevauchement avec la voix précédente : on additionne
                fusion = np.zeros(max(tampon.size, debut - tampon_debut + voix.size), dtype=np.float32)
                fusion[: tampon.size] = tampon
                fusion[debut - tampon_debut: debut - tampon_debut + voix.size] += voix
                tampon = fusion
            else:
                tampon_debut, tampon = debut, voix
        ecrire_jusqua(total)
    return sortie


MARGE_PAROLE_S = 0.25  # autour d'une phrase d'origine, le résidu reste coupé (attaques, fins de mots)
FONDU_RESIDU_S = 0.08


def residu_hors_parole(voix_isolee: Path, phrases: list[tuple[float, float]], sortie: Path) -> Path:
    """Ce que la séparation a pris pour de la voix là où personne ne parle (rires, cris, bruitages,
    ambiance) : la piste « voix isolée », coupée pendant chaque phrase d'origine (avec fondus).

    Remixé avec le fond, il rend au doublage les sons d'origine que la séparation avait retirés, sans
    jamais laisser entendre la voix d'origine sous la voix doublée. Lecture par blocs : mémoire bornée."""
    with sf.SoundFile(voix_isolee) as entree:
        taux, canaux = entree.samplerate, entree.channels
        intervalles = sorted((max(0.0, a - MARGE_PAROLE_S), b + MARGE_PAROLE_S) for a, b in phrases)
        fondu = max(1, int(FONDU_RESIDU_S * taux))
        with sf.SoundFile(sortie, "w", samplerate=taux, channels=canaux, subtype="PCM_16") as fichier:
            position = 0
            for bloc in entree.blocks(blocksize=taux * 10, dtype="float32", always_2d=True):
                t = (position + np.arange(len(bloc))) / taux
                ouvert = np.ones(len(bloc), dtype=np.float32)
                for a, b in intervalles:
                    if b < t[0] - FONDU_RESIDU_S or a > t[-1] + FONDU_RESIDU_S:
                        continue
                    # 1 loin de la phrase, 0 pendant, rampes de FONDU_RESIDU_S de part et d'autre
                    distance = np.maximum(a - t, t - b)
                    ouvert = np.minimum(ouvert, np.clip(distance * taux / fondu, 0.0, 1.0))
                fichier.write(bloc * ouvert[:, None])
                position += len(bloc)
    return sortie


def mixer_fond(voix: Path, fond: Path, volume: int, sortie: Path, residu: Path | None = None) -> Path:
    """Mixe la voix doublée (mono) avec le fond sonore d'origine (stéréo) au volume choisi (0 à 100 %).
    `residu` (residu_hors_parole) complète le fond des sons que la séparation avait pris pour de la voix."""
    niveau = f"{max(0, min(volume, 100)) / 100:.2f}"
    entrees = ["-i", str(voix), "-i", str(fond)]
    if residu is not None:
        entrees += ["-i", str(residu)]
        filtre = (
            "[0:a]aresample=44100,pan=stereo|c0=c0|c1=c0[v];"
            f"[1:a][2:a]amix=inputs=2:duration=first:normalize=0,volume={niveau}[f];"
            "[v][f]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[m]"
        )
    else:
        filtre = (
            "[0:a]aresample=44100,pan=stereo|c0=c0|c1=c0[v];"
            f"[1:a]volume={niveau}[f];"
            "[v][f]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[m]"
        )
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", *entrees,
         "-filter_complex", filtre, "-map", "[m]", "-c:a", "pcm_s16le", str(sortie)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError(f"Le mixage avec le fond sonore a échoué : {resultat.stderr.strip()[:300]}")
    return sortie


def _horodatage_srt(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def ecrire_srt(segments: list[dict[str, Any]], sortie: Path) -> Path:
    """Sous-titres dans la langue cible, calés sur la durée réelle des voix doublées."""
    lignes = []
    for n, seg in enumerate(segments, start=1):
        fin = seg["debut"] + seg.get("duree_doublage", seg["fin"] - seg["debut"])
        lignes += [str(n), f"{_horodatage_srt(seg['debut'])} --> {_horodatage_srt(fin)}", seg["traduction"], ""]
    sortie.write_text("\n".join(lignes), encoding="utf-8")
    return sortie


def exporter_mp3(piste: Path, sortie: Path) -> Path:
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(piste), "-c:a", "libmp3lame", "-q:a", "3", str(sortie)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("La création du MP3 a échoué.")
    return sortie
