"""La vidéo doublée garde exactement le format de la vidéo fournie (conteneur, codecs, résolution,
cadence, son), y compris après « Raccourcir la vidéo » et après la synchronisation labiale."""

import subprocess

import pytest

from pipeline.extract import _require
from pipeline.format import apercu_navigateur, conformer, couper_meme_format, exporter_meme_format, sonder

# extension, codec vidéo, codec audio, largeur, hauteur, cadence, fréquence, canaux
FORMATS = [
    (".mov", "libx264", "aac", 1080, 1920, "30", 48000, 2),  # vidéo verticale de téléphone
    (".mkv", "libx265", "ac3", 1280, 720, "24000/1001", 48000, 6),
    (".webm", "libvpx-vp9", "libopus", 640, 360, "25", 48000, 1),
    (".avi", "mpeg4", "libmp3lame", 720, 576, "25", 44100, 2),
    (".wmv", "wmv2", "wmav2", 320, 240, "15", 22050, 1),
]


def _video(dossier, ext, cv, ca, l, h, r, ar, ac):
    chemin = dossier / f"source{ext}"
    subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={l}x{h}:rate={r}:duration=3",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={ar}:duration=3", "-ac", str(ac),
         "-c:v", cv, "-pix_fmt", "yuv420p", "-c:a", ca, str(chemin)],
        check=True, capture_output=True,
    )
    return chemin


def _piste(dossier):
    chemin = dossier / "doublage.wav"
    subprocess.run([_require("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                    "-ar", "24000", "-ac", "1", str(chemin)], check=True, capture_output=True)
    return chemin


def _signature(fmt):
    return (fmt.extension, fmt.codec_video, fmt.largeur, fmt.hauteur, fmt.cadence, fmt.pixels,
            fmt.codec_audio, fmt.frequence, fmt.canaux)


@pytest.mark.parametrize("ext,cv,ca,l,h,r,ar,ac", FORMATS, ids=[f[0] for f in FORMATS])
def test_export_et_decoupe_gardent_le_format(tmp_path, ext, cv, ca, l, h, r, ar, ac):
    source = _video(tmp_path, ext, cv, ca, l, h, r, ar, ac)
    attendu = _signature(sonder(source))

    doublee = exporter_meme_format(source, _piste(tmp_path), tmp_path / f"doublage{ext}")
    assert _signature(sonder(doublee)) == attendu

    coupee = couper_meme_format(source, tmp_path / f"coupee{ext}", 0.5, 2.5)
    assert _signature(sonder(coupee)) == attendu


def test_synchro_remise_au_format_d_origine(tmp_path):
    source = _video(tmp_path, ".mov", "libx264", "aac", 1080, 1920, "30", 48000, 2)
    # Ce que renvoie la synchronisation labiale : autre taille, 25 images/s, MP4
    kaggle = tmp_path / "synchro.mp4"
    subprocess.run([_require("ffmpeg"), "-y", "-v", "error", "-i", str(source), "-vf", "scale=512:910", "-r", "25",
                    "-c:v", "libx264", "-c:a", "aac", "-ar", "44100", "-ac", "1", str(kaggle)],
                   check=True, capture_output=True)
    remise = conformer(kaggle, source, tmp_path / "doublage_synchro.mov")
    assert _signature(sonder(remise)) == _signature(sonder(source))


def test_apercu_seulement_si_le_navigateur_ne_lit_pas(tmp_path):
    mp4 = _video(tmp_path, ".mp4", "libx264", "aac", 320, 240, "25", 44100, 2)
    assert apercu_navigateur(mp4, tmp_path / "apercu_mp4.mp4") is None
    avi = _video(tmp_path, ".avi", "mpeg4", "libmp3lame", 320, 240, "25", 44100, 2)
    apercu = apercu_navigateur(avi, tmp_path / "apercu.mp4")
    assert apercu is not None and sonder(apercu).codec_video == "h264"
