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


def _sous_titre_sur_fond_texture():
    """Fond à motifs (comme nourriture, fourrure, foule) avec un sous-titre blanc contouré de noir."""
    import cv2
    import numpy as np

    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w]
    fond = np.zeros((h, w, 3), np.uint8)
    fond[..., 0] = 90 + 60 * np.sin(xx / 6.0) * np.cos(yy / 9.0)
    fond[..., 1] = 120 + 50 * np.sin((xx + yy) / 7.0)
    fond[..., 2] = 150 + 70 * np.cos(xx / 11.0)
    fond = cv2.add(fond, np.random.default_rng(0).integers(0, 25, fond.shape, dtype=np.uint8))
    image = fond.copy()
    cv2.putText(image, "Bonjour tout le monde", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(image, "Bonjour tout le monde", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    masque = cv2.dilate((np.abs(image.astype(int) - fond).max(axis=2) > 0).astype(np.uint8) * 255,
                        np.ones((5, 5), np.uint8))
    return fond, image, masque


def _ecarts_lama_telea(modele):
    """Écart moyen au vrai fond, sous le texte, après reconstruction par LaMa puis par Telea, et durées (s)."""
    import time

    import cv2
    import numpy as np

    from pipeline.habillage import _lama

    fond, image, masque = _sous_titre_sur_fond_texture()
    a, b, g, d = 226, 344, 16, 624  # rectangle des lignes, marge de contexte comprise
    garde = masque[a:b, g:d] > 0
    _lama(modele, image[a:b, g:d], masque[a:b, g:d])  # premier passage : préparation du réseau
    debut = time.perf_counter()
    lama = _lama(modele, image[a:b, g:d], masque[a:b, g:d])
    duree_lama = time.perf_counter() - debut
    debut = time.perf_counter()
    telea = cv2.inpaint(image[a:b, g:d], masque[a:b, g:d], 3, cv2.INPAINT_TELEA)
    duree_telea = time.perf_counter() - debut
    vrai = fond[a:b, g:d].astype(int)
    return (np.abs(lama.astype(int) - vrai)[garde].mean(), np.abs(telea.astype(int) - vrai)[garde].mean(),
            duree_lama, duree_telea)


def test_effacement_sur_processeur_reste_telea(monkeypatch):
    """Sans carte graphique, LaMa n'est ni téléchargé ni chargé : l'effacement reste celui de Telea."""
    import config
    from pipeline import habillage

    monkeypatch.setattr(config, "get_runtime", lambda: type("R", (), {"device": "cpu"})())
    habillage._retoucheur.cache_clear()
    try:
        assert habillage._retoucheur() is None
    finally:
        habillage._retoucheur.cache_clear()


def test_lama_ne_change_que_les_pixels_masques(monkeypatch):
    """Le résultat du réseau n'est recopié que sur les pixels des lettres : le décor voisin reste intact."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from pipeline import habillage

    class Gris(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.poids = torch.nn.Parameter(torch.zeros(1))  # situe le modèle (processeur ici)

        def forward(self, x, m):
            assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0
            return torch.full_like(x, 0.5)

    monkeypatch.setattr(habillage, "_retoucheur", lambda: Gris())
    monkeypatch.setattr(habillage, "_LAMA_EN_PANNE", [])
    roi = np.full((50, 70, 3), 200, np.uint8)
    masque = np.zeros((50, 70), np.uint8)
    masque[20:30, 10:60] = 255
    habillage._reconstruire(roi, masque, 15, 35, 5, 65)
    assert (roi[20:30, 10:60] == 128).all()
    roi[20:30, 10:60] = 200
    assert (roi == 200).all()


def test_lama_en_echec_repli_sur_telea(monkeypatch):
    """Si LaMa échoue (mémoire graphique saturée…), le texte est quand même effacé, par Telea."""
    import cv2
    import numpy as np

    from pipeline import habillage

    def panne(*_):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(habillage, "_retoucheur", lambda: object())
    monkeypatch.setattr(habillage, "_lama", panne)
    monkeypatch.setattr(habillage, "_LAMA_EN_PANNE", [])
    image = np.full((360, 640, 3), (180, 120, 60), np.uint8)
    cv2.putText(image, "Bonjour tout le monde", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(image, "Bonjour tout le monde", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    habillage.effacer_texte(image, (20, 240, 620, 340))
    assert habillage._LAMA_EN_PANNE  # les images suivantes passent directement par Telea
    assert np.abs(image[250:330, 40:600].astype(int) - (180, 120, 60)).max(axis=2).mean() < 12


def test_lama_plus_propre_que_telea_sur_fond_texture():
    """Avec le vrai modèle (déjà téléchargé), LaMa reconstruit mieux un décor à motifs que Telea.
    Sur processeur, LaMa prend plusieurs secondes : le test ne tourne que si le modèle est présent."""
    torch = pytest.importorskip("torch")
    from config import get_settings

    chemin = get_settings().models_dir / "texte" / "big-lama.pt"
    if not chemin.is_file():
        pytest.skip("modèle LaMa absent (téléchargé au premier effacement sur carte graphique)")
    modele = torch.jit.load(str(chemin), map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    ecart_lama, ecart_telea, _, _ = _ecarts_lama_telea(modele)
    assert ecart_lama < 0.85 * ecart_telea


@pytest.mark.skipif("not __import__('importlib').util.find_spec('torch') "
                    "or not __import__('torch').cuda.is_available()", reason="pas de carte graphique NVIDIA")
def test_effacement_lama_sur_gpu(capsys):
    """Sur carte graphique, l'effacement passe par LaMa : plus propre que Telea, en un temps raisonnable."""
    from pipeline import habillage

    habillage._retoucheur.cache_clear()
    modele = habillage._retoucheur()
    assert modele is not None
    ecart_lama, ecart_telea, duree_lama, duree_telea = _ecarts_lama_telea(modele)
    with capsys.disabled():
        print(f"\nLaMa (GPU) : {duree_lama * 1000:.0f} ms, écart {ecart_lama:.1f} — "
              f"Telea : {duree_telea * 1000:.0f} ms, écart {ecart_telea:.1f}")
    assert ecart_lama < 0.85 * ecart_telea
    assert duree_lama < 0.5
    fond, image, _ = _sous_titre_sur_fond_texture()
    habillage.effacer_texte(image, (20, 240, 620, 340))
    assert not habillage._LAMA_EN_PANNE
