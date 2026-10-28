"""Habillage de la vidéo exportée : sous-titres incrustés, flou d'une zone, logo, qualité, métadonnées.
La vidéo habillée garde le conteneur et le codec de la vidéo doublée ; seules les options demandées changent."""

import json
import subprocess

import pytest

from pipeline.extract import _require
from pipeline.format import sonder
from pipeline.habillage import Taille, _blocs, _mots_minutes, actif, apercu, ecrire_ass, habiller, normaliser

SEGMENTS = [
    {"id": 0, "debut": 0.4, "fin": 1.6, "traduction": "Bonjour {à} tous, voici VoxLingo !", "duree_doublage": 1.2},
    {"id": 1, "debut": 1.8, "fin": 2.8, "traduction": "Les mots s'animent.", "duree_doublage": 0.9},
]


def _video(dossier, ext=".mp4", cv="libx264", ca="aac", taille="640x360"):
    chemin = dossier / f"doublage{ext}"
    subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={taille}:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", cv, "-pix_fmt", "yuv420p", "-c:a", ca,
         "-metadata", "title=Titre secret", "-metadata", "comment=Téléphone de Jean", str(chemin)],
        check=True, capture_output=True,
    )
    return chemin


def _etiquettes(video):
    sortie = subprocess.run([_require("ffprobe"), "-v", "error", "-show_entries", "format_tags:stream_tags",
                             "-of", "json", str(video)], capture_output=True, text=True).stdout
    donnees = json.loads(sortie)
    return json.dumps(donnees, ensure_ascii=False)


def test_reglages_normalises():
    r = normaliser({"style": "inconnu", "couleur": "rouge", "zone": {"x": 0.9, "l": 0.5}})
    assert r["style"] == "reseaux" and r["couleur"] == "#facc15"
    assert r["zone"]["x"] + r["zone"]["l"] <= 1.0
    assert not actif({}) and not actif(None)
    assert actif({"sans_metadonnees": True})


def test_mots_repartis_sur_la_voix_doublee():
    mots = _mots_minutes(SEGMENTS[0])
    assert mots[0][1] == pytest.approx(0.4) and mots[-1][2] == pytest.approx(1.6)
    assert all(a[2] == pytest.approx(b[1]) for a, b in zip(mots, mots[1:]))
    # nouveau groupe après une fin de phrase, et jamais plus de mots que le style n'en affiche
    assert all(len(b) <= 3 for b in _blocs(mots, 3))


@pytest.mark.parametrize("style", ["reseaux", "impact", "karaoke", "classique"])
def test_fichier_ass(tmp_path, style):
    r = normaliser({"sous_titres": True, "style": style, "animation": "glisse", "logo": True})
    texte = ecrire_ass(SEGMENTS, r, Taille(1080, 1920), 3.0, tmp_path / "h.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in texte and "PlayResY: 1920" in texte
    assert "{à}" not in texte  # accolades du texte neutralisées
    assert texte.count("Style: Logo") == 1 and "VoxLingo" in texte
    if style == "karaoke":
        assert "\\kf" in texte
    if style == "impact":
        assert "BONJOUR" in texte


def test_habillage_complet_meme_format_sans_metadonnees(tmp_path):
    source = _video(tmp_path)
    r = {"sous_titres": True, "flou": True, "logo": True, "qualite": True, "sans_metadonnees": True}
    progression = []
    sortie = habiller(source, SEGMENTS, r, tmp_path / "habille.mp4", progression.append)
    fmt = sonder(sortie)
    assert (fmt.extension, fmt.codec_video, fmt.codec_audio) == (".mp4", "h264", "aac")
    assert (fmt.largeur, fmt.hauteur, fmt.cadence) == (640, 360, "25/1")  # format inchangé
    assert progression[-1] == 1.0
    etiquettes = _etiquettes(sortie)
    assert "secret" not in etiquettes and "Jean" not in etiquettes and "Lavf" not in etiquettes
    assert b"x264 - core" not in sortie.read_bytes()
    assert not (tmp_path / "habillage.ass").exists()


def test_metadonnees_seules_sans_reencodage(tmp_path):
    source = _video(tmp_path, ".mkv", "libx265", "libopus")
    sortie = habiller(source, SEGMENTS, {"sans_metadonnees": True}, tmp_path / "habille.mkv")
    assert (sonder(sortie).codec_video, sonder(sortie).codec_audio) == ("hevc", "opus")
    assert "secret" not in _etiquettes(sortie)


def test_metadonnees_conservees_si_non_demande(tmp_path):
    source = _video(tmp_path)
    sortie = habiller(source, SEGMENTS, {"logo": True}, tmp_path / "habille.mp4")
    assert "Titre secret" in _etiquettes(sortie)


def test_interruption(tmp_path):
    source = _video(tmp_path)
    with pytest.raises(InterruptedError):
        habiller(source, SEGMENTS, {"qualite": True}, tmp_path / "habille.mp4", doit_arreter=lambda: True)
    assert not (tmp_path / "habille.mp4").exists()


def test_apercu_jpeg(tmp_path):
    source = _video(tmp_path)
    image = apercu(source, SEGMENTS, {"sous_titres": True, "flou": True, "logo": True}, tmp_path, None)
    assert image[:2] == b"\xff\xd8"
    assert not list(tmp_path.glob("apercu_*.ass"))


def test_effacement_ne_touche_pas_une_image_sans_texte(tmp_path):
    """Sans sous-titre dans la zone, l'image ressort identique, pixel pour pixel."""
    import numpy as np

    from pipeline.habillage import effacer_texte

    image = np.zeros((360, 640, 3), np.uint8)
    image[:] = np.linspace(40, 200, 640, dtype=np.uint8)[None, :, None]  # dégradé, sans texte
    avant = image.copy()
    effacer_texte(image, (40, 260, 600, 340))
    assert np.array_equal(image, avant)


def test_effacement_retire_le_texte_seulement_dans_la_zone(tmp_path):
    import cv2
    import numpy as np

    from pipeline.habillage import effacer_texte

    image = np.full((360, 640, 3), (180, 120, 60), np.uint8)  # fond bleu uni
    for y in (60, 300):  # même texte dans la zone (en bas) et hors zone (en haut)
        cv2.putText(image, "Bonjour tout le monde", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(image, "Bonjour tout le monde", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    haut = image[:200].copy()
    effacer_texte(image, (20, 240, 620, 340))
    assert np.array_equal(image[:200], haut)  # hors zone : intact
    fond = np.array((180, 120, 60))
    assert np.abs(image[250:330, 40:600].astype(int) - fond).max(axis=2).mean() < 12  # texte effacé
