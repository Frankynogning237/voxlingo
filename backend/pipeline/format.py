"""Format de la vidéo fournie, conservé à l'identique dans tout ce que VoxLingo produit.

La vidéo doublée garde le conteneur (MP4, MOV, MKV, WEBM, AVI…), le codec, la résolution, la
cadence et le format de pixels de l'original, ainsi que le codec, la fréquence, le nombre de
canaux et le débit de son audio. L'image est copiée telle quelle quand c'est possible ; quand
elle doit être réencodée (vidéo raccourcie, vidéo synchronisée revenant de Kaggle), elle l'est
avec le même codec, à la même taille et à la même cadence, en haute qualité.

Seul l'aperçu dans le navigateur peut être une copie MP4 (H.264), si le format d'origine ne
s'y lit pas (AVI, WMV, MKV en HEVC…) : le fichier téléchargé, lui, garde toujours le format fourni.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline.extract import FFmpegError, _require

logger = logging.getLogger("voxlingo.format")

# Encodeur FFmpeg de chaque codec vidéo (réencodage dans le même codec)
ENCODEURS_VIDEO = {
    "h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9", "vp8": "libvpx", "av1": "libsvtav1",
    "mpeg4": "mpeg4", "msmpeg4v3": "msmpeg4", "wmv1": "wmv1", "wmv2": "wmv2", "mpeg2video": "mpeg2video",
    "mpeg1video": "mpeg1video", "flv1": "flv", "mjpeg": "mjpeg", "prores": "prores_ks", "theora": "libtheora",
}
# Encodeur FFmpeg de chaque codec audio
ENCODEURS_AUDIO = {
    "aac": "aac", "mp3": "libmp3lame", "opus": "libopus", "vorbis": "libvorbis", "ac3": "ac3", "eac3": "eac3",
    "mp2": "mp2", "flac": "flac", "alac": "alac", "wmav2": "wmav2", "pcm_s16le": "pcm_s16le",
}
# Vidéo sans piste audio d'origine : codec audio habituel du conteneur
AUDIO_PAR_CONTENEUR = {
    ".webm": "libopus", ".avi": "libmp3lame", ".wmv": "wmav2", ".mpg": "mp2", ".mpeg": "mp2",
}
SANS_DEBIT = {"flac", "alac", "pcm_s16le"}  # codecs sans débit à fixer (sans perte)
MOV_FAMILLE = {".mp4", ".m4v", ".mov"}  # conteneurs qui acceptent « faststart » (lecture immédiate en ligne)
LISIBLES_NAVIGATEUR = {  # conteneur → codecs vidéo lus par les navigateurs
    ".mp4": {"h264", "vp9", "av1"}, ".m4v": {"h264"}, ".mov": {"h264"}, ".webm": {"vp8", "vp9", "av1"},
}


@dataclass(frozen=True)
class Format:
    extension: str  # « .mp4 », « .mkv »…
    codec_video: str | None
    largeur: int | None
    hauteur: int | None
    cadence: str | None  # « 30000/1001 », « 25/1 »…
    pixels: str | None  # format de pixels (« yuv420p »…)
    debit_video: int | None
    codec_audio: str | None
    frequence: int | None
    canaux: int | None
    debit_audio: int | None

    @property
    def lisible_navigateur(self) -> bool:
        return self.codec_video in LISIBLES_NAVIGATEUR.get(self.extension, set())


@lru_cache(maxsize=1)
def _encodeurs_disponibles() -> frozenset[str]:
    sortie = subprocess.run([_require("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return frozenset(ligne.split()[1] for ligne in sortie.splitlines() if len(ligne.split()) > 1 and ligne[:1] == " ")


def sonder(video: Path) -> Format:
    """Lit le format exact d'une vidéo (ffprobe)."""
    resultat = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("Impossible de lire le format de la vidéo.")
    donnees = json.loads(resultat.stdout)
    flux = donnees.get("streams", [])
    v = next((f for f in flux if f.get("codec_type") == "video" and not (f.get("disposition") or {}).get("attached_pic")), {})
    a = next((f for f in flux if f.get("codec_type") == "audio"), {})

    def entier(valeur) -> int | None:
        try:
            return int(valeur)
        except (TypeError, ValueError):
            return None

    return Format(
        extension=video.suffix.lower(),
        codec_video=v.get("codec_name"),
        largeur=entier(v.get("width")),
        hauteur=entier(v.get("height")),
        cadence=v.get("r_frame_rate") if v.get("r_frame_rate") not in (None, "0/0") else None,
        pixels=v.get("pix_fmt"),
        debit_video=entier(v.get("bit_rate")),
        codec_audio=a.get("codec_name"),
        frequence=entier(a.get("sample_rate")),
        canaux=entier(a.get("channels")),
        debit_audio=entier(a.get("bit_rate")),
    )


def arguments_audio(fmt: Format) -> list[str]:
    """Encodage du son doublé comme celui de l'original (codec, fréquence, canaux, débit)."""
    encodeur = ENCODEURS_AUDIO.get(fmt.codec_audio or "") or AUDIO_PAR_CONTENEUR.get(fmt.extension, "aac")
    if encodeur not in _encodeurs_disponibles():
        logger.warning("Encodeur audio %s indisponible : AAC", encodeur)
        encodeur = "aac"
    arguments = ["-c:a", encodeur]
    if fmt.frequence:
        arguments += ["-ar", str(fmt.frequence)]
    if fmt.canaux:
        arguments += ["-ac", str(fmt.canaux)]
    if encodeur not in SANS_DEBIT:
        arguments += ["-b:a", str(fmt.debit_audio or 192_000)]
    return arguments


def arguments_video(fmt: Format) -> list[str]:
    """Réencodage de l'image dans le même codec, à la même taille, cadence et format de pixels."""
    encodeur = ENCODEURS_VIDEO.get(fmt.codec_video or "", "libx264")
    if encodeur not in _encodeurs_disponibles():
        logger.warning("Encodeur vidéo %s indisponible : H.264", encodeur)
        encodeur = "libx264"
    arguments = ["-c:v", encodeur]
    if encodeur in ("libx264", "libx265"):
        arguments += ["-crf", "16", "-preset", "medium"]
    elif encodeur == "libvpx-vp9":
        arguments += ["-crf", "20", "-b:v", "0", "-row-mt", "1"]
    elif encodeur == "libsvtav1":
        arguments += ["-crf", "24", "-preset", "8"]
    elif fmt.debit_video:
        arguments += ["-b:v", str(int(fmt.debit_video * 1.2))]  # un peu plus que l'original : pas de perte visible
    else:
        arguments += ["-q:v", "2"]
    if fmt.pixels:
        arguments += ["-pix_fmt", fmt.pixels]
    if fmt.cadence:
        arguments += ["-r", fmt.cadence]
    return arguments


def _conteneur(fmt: Format) -> list[str]:
    return ["-movflags", "+faststart"] if fmt.extension in MOV_FAMILLE else []


def _executer(commande: list[str], message: str) -> None:
    resultat = subprocess.run(commande, capture_output=True, text=True)
    if resultat.returncode != 0:
        raise FFmpegError(f"{message} : {resultat.stderr.strip()[-300:]}")


def exporter_meme_format(video: Path, piste: Path, sortie: Path) -> Path:
    """Vidéo doublée au format exact de `video` : image copiée sans réencodage (mêmes codec,
    résolution et cadence), son doublé encodé comme le son d'origine, même conteneur.
    `sortie` doit porter la même extension que `video`."""
    fmt = sonder(video)
    _executer(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-i", str(piste),
         "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
         *arguments_audio(fmt), *_conteneur(fmt), str(sortie)],
        "L'export de la vidéo doublée a échoué",
    )
    return sortie


def couper_meme_format(video: Path, sortie: Path, debut_s: float, fin_s: float) -> Path:
    """Garde le passage [debut_s, fin_s] à l'image près, dans le même format que l'original
    (même conteneur, codecs, résolution, cadence)."""
    fmt = sonder(video)
    _executer(
        [_require("ffmpeg"), "-y", "-v", "error", "-noautorotate", "-ss", f"{debut_s:.3f}", "-i", str(video),
         "-t", f"{fin_s - debut_s:.3f}", "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0",
         *arguments_video(fmt), *arguments_audio(fmt), *_conteneur(fmt), str(sortie)],
        "Impossible de raccourcir la vidéo",
    )
    return sortie


def conformer(video: Path, reference: Path, sortie: Path) -> Path:
    """Remet une vidéo produite ailleurs (synchronisation labiale sur Kaggle : 25 images/s, autre
    taille…) au format exact de `reference` : même résolution, cadence, codecs et conteneur."""
    ref, fmt = sonder(reference), sonder(video)
    filtres = []
    if ref.largeur and ref.hauteur and (fmt.largeur, fmt.hauteur) != (ref.largeur, ref.hauteur):
        filtres.append(f"scale={ref.largeur}:{ref.hauteur}:flags=lanczos")
    commande = [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?"]
    if filtres:
        commande += ["-vf", ",".join(filtres)]
    _executer([*commande, *arguments_video(ref), *arguments_audio(ref), *_conteneur(ref), str(sortie)],
              "La mise au format d'origine a échoué")
    return sortie


def apercu_navigateur(video: Path, sortie: Path, toujours: bool = False) -> Path | None:
    """Copie MP4 (H.264) pour l'aperçu dans le navigateur, seulement si le format d'origine ne s'y
    lit pas (ou `toujours`). Le fichier téléchargeable reste au format d'origine."""
    fmt = sonder(video)
    if fmt.lisible_navigateur and not toujours:
        sortie.unlink(missing_ok=True)
        return None
    _executer(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(sortie)],
        "La création de l'aperçu a échoué",
    )
    return sortie
