"""Habillage facultatif de la vidéo exportée : sous-titres traduits incrustés, effacement par flou des
anciens sous-titres (lettres détectées puis reconstruites), petit logo VoxLingo, légère amélioration de l'image, suppression des métadonnées.
Le format de la vidéo (résolution, cadence, codec, conteneur) n'est jamais modifié.

L'habillage s'applique sur la vidéo finale (doublée, ou synchronisée si la synchronisation labiale a été
faite) et produit `habille.<ext>`, au même conteneur et au même codec. Il se refait seul, sans régénérer
les voix. Les sous-titres et le logo sont écrits dans un fichier ASS rendu par libass (filtre `ass`) :
le traitement est fait image par image par FFmpeg, en mémoire bornée quelle que soit la durée.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline.extract import FFmpegError, _require, probe_duration
from pipeline.format import _conteneur, arguments_video, sonder

logger = logging.getLogger("voxlingo.habillage")

STYLES = ("reseaux", "impact", "karaoke", "classique")
ANIMATIONS = ("aucune", "fondu", "pop", "glisse")
TAILLES = {"petite": 0.8, "moyenne": 1.0, "grande": 1.25}
POSITIONS = ("bas", "milieu", "haut")
COINS = ("haut_droite", "haut_gauche", "bas_droite", "bas_gauche")

PAR_DEFAUT: dict[str, Any] = {
    "sous_titres": False, "style": "reseaux", "animation": "pop", "taille": "moyenne", "position": "bas",
    "couleur": "#facc15",
    "flou": False, "zone": {"x": 0.08, "y": 0.76, "l": 0.84, "h": 0.2},
    "logo": False, "coin": "haut_droite",
    "qualite": False,
    "sans_metadonnees": False,
}

# Par style : police, taille (fraction du petit côté de l'image), contour, ombre, gras, mots par écran
_STYLES = {
    "classique": {"police": "Arial", "taille": 0.055, "contour": 0.0035, "ombre": 0.002, "gras": 0, "mots": 12},
    "reseaux": {"police": "Arial", "taille": 0.07, "contour": 0.007, "ombre": 0.0, "gras": -1, "mots": 3},
    "impact": {"police": "Impact", "taille": 0.085, "contour": 0.008, "ombre": 0.003, "gras": -1, "mots": 2},
    "karaoke": {"police": "Arial", "taille": 0.062, "contour": 0.005, "ombre": 0.002, "gras": -1, "mots": 8},
}
_LOGO = [(2, 14), (8, 26), (14, 38), (20, 26), (26, 34)]  # barres du logo (x, hauteur) sur 30 × 40
_DEGRADE_LOGO = [(0.0, "#c084fc"), (0.55, "#8b5cf6"), (1.0, "#22d3ee")]


def normaliser(reglages: dict[str, Any] | None) -> dict[str, Any]:
    """Réglages complets et valides (valeurs par défaut pour ce qui manque ou est inconnu)."""
    r = {**PAR_DEFAUT, **(reglages or {})}
    r["zone"] = {**PAR_DEFAUT["zone"], **(r.get("zone") or {})}
    for cle, valeurs in (("style", STYLES), ("animation", ANIMATIONS), ("taille", TAILLES),
                         ("position", POSITIONS), ("coin", COINS)):
        if r[cle] not in valeurs:
            r[cle] = PAR_DEFAUT[cle]
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(r["couleur"])):
        r["couleur"] = PAR_DEFAUT["couleur"]
    z = r["zone"]
    for cle in ("x", "y", "l", "h"):
        z[cle] = min(1.0, max(0.0, float(z[cle])))
    z["l"], z["h"] = min(z["l"], 1 - z["x"]), min(z["h"], 1 - z["y"])
    return {cle: r[cle] for cle in PAR_DEFAUT}


def actif(reglages: dict[str, Any] | None) -> bool:
    """Au moins un élément d'habillage est demandé."""
    r = normaliser(reglages)
    return any(r[c] for c in ("sous_titres", "flou", "logo", "qualite", "sans_metadonnees"))


def _visuel(r: dict[str, Any]) -> bool:
    """L'image doit être réencodée (sinon seules les métadonnées changent : copie sans perte)."""
    return any(r[c] for c in ("sous_titres", "flou", "logo", "qualite"))


# ---------------------------------------------------------------------------------------------------
# Taille de l'image
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Taille:
    largeur: int
    hauteur: int

    @property
    def ref(self) -> int:
        """Petit côté : sert d'échelle aux textes (même rendu en vidéo verticale ou horizontale)."""
        return min(self.largeur, self.hauteur)


def taille_affichee(video: Path) -> Taille:
    """Largeur et hauteur de l'image telle qu'elle s'affiche (rotation des vidéos de téléphone comprise)."""
    resultat = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation:stream_tags=rotate", "-of", "json", str(video)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("Impossible de lire la taille de la vidéo.")
    flux = (json.loads(resultat.stdout).get("streams") or [{}])[0]
    largeur, hauteur = int(flux.get("width") or 0), int(flux.get("height") or 0)
    rotation = (flux.get("tags") or {}).get("rotate") or next(
        (d.get("rotation") for d in flux.get("side_data_list") or [] if "rotation" in d), 0)
    if abs(int(float(rotation or 0))) % 180 == 90:
        largeur, hauteur = hauteur, largeur
    if not largeur or not hauteur:
        raise FFmpegError("Taille de la vidéo inconnue.")
    return Taille(largeur, hauteur)


# ---------------------------------------------------------------------------------------------------
# Sous-titres et logo (ASS)
# ---------------------------------------------------------------------------------------------------

def _couleur_ass(hexa: str, alpha: int = 0) -> str:
    """« #rrggbb » → « &HAABBGGRR » (alpha 0 = opaque, 255 = transparent)."""
    rr, gg, bb = hexa[1:3], hexa[3:5], hexa[5:7]
    return f"&H{alpha:02X}{bb}{gg}{rr}".upper()


def _melange(stops: list[tuple[float, str]], t: float) -> str:
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0) if t1 > t0 else 0
            canaux = [round(int(c0[i:i + 2], 16) * (1 - k) + int(c1[i:i + 2], 16) * k) for i in (1, 3, 5)]
            return "#" + "".join(f"{c:02x}" for c in canaux)
    return stops[-1][1]


def _temps(s: float) -> str:
    cs = max(0, int(round(s * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    sec, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _texte_ass(texte: str) -> str:
    """Texte sûr pour ASS : ni accolades (blocs de commandes) ni barre oblique inverse (échappements)."""
    return texte.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _mots_minutes(seg: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Mots de la traduction, chacun avec son début et sa fin, répartis sur la durée réelle de la voix
    doublée au prorata de leur longueur (la voix doublée n'a pas de minutage au mot)."""
    mots = seg.get("traduction", "").split()
    if not mots:
        return []
    debut = float(seg["debut"])
    duree = float(seg.get("duree_doublage") or (seg["fin"] - seg["debut"]))
    poids = [len(m) + 1 for m in mots]
    total, cumul, sortie = sum(poids), 0, []
    for mot, p in zip(mots, poids):
        sortie.append((mot, debut + duree * cumul / total, debut + duree * (cumul + p) / total))
        cumul += p
    return sortie


def _blocs(mots: list[tuple[str, float, float]], maximum: int) -> list[list[tuple[str, float, float]]]:
    """Groupes de mots affichés ensemble : au plus `maximum`, et un nouveau groupe après une fin de phrase."""
    blocs: list[list[tuple[str, float, float]]] = [[]]
    for mot in mots:
        if len(blocs[-1]) >= maximum:
            blocs.append([])
        blocs[-1].append(mot)
        if re.search(r"[.!?;:…]$", mot[0]) and len(blocs[-1]) >= min(maximum, 3):
            blocs.append([])
    return [b for b in blocs if b]


def _ancre(r: dict[str, Any], t: Taille) -> tuple[int, int, int]:
    """Alignement ASS et point d'ancrage du texte selon la position choisie."""
    marge = int(t.ref * 0.07)
    return {
        "bas": (2, t.largeur // 2, t.hauteur - marge),
        "milieu": (5, t.largeur // 2, t.hauteur // 2),
        "haut": (8, t.largeur // 2, marge),
    }[r["position"]]


def _entree(r: dict[str, Any], t: Taille, premier: bool, dernier: bool) -> str:
    """Commandes d'animation d'un événement (entrée sur le premier, sortie sur le dernier d'un groupe)."""
    an, x, y = _ancre(r, t)
    tags = f"\\an{an}"
    anim = r["animation"]
    if anim == "fondu":
        tags += f"\\fad({150 if premier else 0},{120 if dernier else 0})"
    elif anim == "pop" and premier:
        tags += "\\fscx70\\fscy70\\t(0,110,\\fscx110\\fscy110)\\t(110,190,\\fscx100\\fscy100)"
    elif anim == "glisse":
        if premier:
            tags += f"\\move({x},{y + int(t.ref * 0.04)},{x},{y},0,180)\\fad(180,0)"
        else:
            tags += f"\\pos({x},{y})"
    return tags


def _evenements_sous_titres(segments: list[dict[str, Any]], r: dict[str, Any], t: Taille) -> list[str]:
    style = r["style"]
    accent = _couleur_ass(r["couleur"])
    lignes = []
    for seg in segments:
        for bloc in _blocs(_mots_minutes(seg), _STYLES[style]["mots"]):
            debut, fin = bloc[0][1], bloc[-1][2]
            mots = [m for m, _, _ in bloc]
            if style == "impact":
                mots = [m.upper() for m in mots]
            if style == "reseaux":
                # Le mot prononcé ressort dans la couleur choisie
                for i, (_, d, f) in enumerate(bloc):
                    texte = " ".join(
                        f"{{\\c{accent}\\fscx106\\fscy106}}{_texte_ass(m)}{{\\r}}" if j == i else _texte_ass(m)
                        for j, m in enumerate(mots))
                    tags = _entree(r, t, i == 0, i == len(bloc) - 1)
                    lignes.append(f"Dialogue: 0,{_temps(d)},{_temps(f if i < len(bloc) - 1 else fin)},ST,,0,0,0,,"
                                  f"{{{tags}}}{texte}")
            elif style == "karaoke":
                # Remplissage progressif de chaque mot (\kf, en centièmes de seconde)
                texte = " ".join(f"{{\\kf{max(1, round((f - d) * 100))}}}{_texte_ass(m)}"
                                 for m, (_, d, f) in zip(mots, bloc))
                lignes.append(f"Dialogue: 0,{_temps(debut)},{_temps(fin)},ST,,0,0,0,,"
                              f"{{{_entree(r, t, True, True)}}}{texte}")
            else:
                lignes.append(f"Dialogue: 0,{_temps(debut)},{_temps(fin)},ST,,0,0,0,,"
                              f"{{{_entree(r, t, True, True)}}}{_texte_ass(' '.join(mots))}")
    return lignes


def _evenements_logo(r: dict[str, Any], t: Taille, duree: float) -> list[str]:
    """Logo discret (barres d'onde en dégradé + « VoxLingo »), semi-transparent, dans le coin choisi."""
    h = max(14, int(t.ref * 0.045))
    s = h / 40
    marge = int(t.ref * 0.03)
    largeur_barres = int(30 * s)
    droite, bas = r["coin"].endswith("droite"), r["coin"].startswith("bas")
    x0 = t.largeur - marge - largeur_barres if droite else marge
    y0 = t.hauteur - marge - h if bas else marge
    fin = _temps(duree + 1)
    commun = "\\bord0\\shad0\\alpha&H59&"
    lignes = []
    for x, hb in _LOGO:
        couleur = _couleur_ass(_melange(_DEGRADE_LOGO, x / 26))
        g, d, hh, b = 0, 3.2 * s, (40 - hb) / 2 * s, (40 + hb) / 2 * s
        dessin = f"m {g:.1f} {hh:.1f} l {d:.1f} {hh:.1f} l {d:.1f} {b:.1f} l {g:.1f} {b:.1f}"
        lignes.append(f"Dialogue: 5,0:00:00.00,{fin},Logo,,0,0,0,,"
                      f"{{\\an7\\pos({x0 + x * s:.1f},{y0}){commun}\\1c{couleur}\\p1}}{dessin}")
    ecart = int(h * 0.25)
    an, xt = (6, x0 - ecart) if droite else (4, x0 + largeur_barres + ecart)
    lignes.append(f"Dialogue: 5,0:00:00.00,{fin},Logo,,0,0,0,,"
                  f"{{\\an{an}\\pos({xt},{y0 + h // 2})\\fs{int(h * 0.62)}\\bord1\\3c&H000000&\\3a&HB0&"
                  f"\\1a&H59&}}VoxLingo")
    return lignes


def ecrire_ass(segments: list[dict[str, Any]], r: dict[str, Any], t: Taille, duree: float, sortie: Path) -> Path:
    """Fichier ASS des sous-titres incrustés et du logo, à la taille exacte de l'image produite."""
    st = _STYLES[r["style"]]
    taille = int(t.ref * st["taille"] * TAILLES[r["taille"]])
    an, _, _ = _ancre(r, t)
    marge_v, marge_h = int(t.ref * 0.07), int(t.ref * 0.06)
    blanc, noir = "&H00FFFFFF", "&H00000000"
    # Karaoké : le mot passe du blanc (couleur secondaire) à la couleur choisie (principale)
    principale = _couleur_ass(r["couleur"]) if r["style"] in ("impact", "karaoke") else blanc
    style = (f"Style: ST,{st['police']},{taille},{principale},{blanc},{noir},&H80000000,{st['gras']},0,0,0,"
             f"100,100,0,0,1,{max(1, round(t.ref * st['contour']))},{round(t.ref * st['ombre'])},"
             f"{an},{marge_h},{marge_h},{marge_v},1")
    logo = f"Style: Logo,Arial,20,{blanc},{blanc},{noir},{noir},-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
    evenements = []
    if r["sous_titres"]:
        evenements += _evenements_sous_titres(segments, r, t)
    if r["logo"]:
        evenements += _evenements_logo(r, t, duree)
    contenu = "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {t.largeur}", f"PlayResY: {t.hauteur}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: None", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        style, logo, "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *evenements, "",
    ])
    sortie.write_text(contenu, encoding="utf-8")
    return sortie


# ---------------------------------------------------------------------------------------------------
# Effacement des anciens sous-titres incrustés
# ---------------------------------------------------------------------------------------------------

def zone_pixels(z: dict[str, float], t: Taille) -> tuple[int, int, int, int]:
    """Zone choisie (fractions de l'image) en pixels : x0, y0, x1, y1."""
    x0, y0 = int(z["x"] * t.largeur), int(z["y"] * t.hauteur)
    x1, y1 = int((z["x"] + z["l"]) * t.largeur), int((z["y"] + z["h"]) * t.hauteur)
    return x0, y0, max(x0 + 4, min(x1, t.largeur)), max(y0 + 4, min(y1, t.hauteur))


URL_DETECTEUR = "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx"


@lru_cache(maxsize=1)
def _detecteur():
    """Détecteur de texte PP-OCRv4 (DB, ~5 Mo, ONNX sur processeur), téléchargé au premier usage."""
    import httpx
    import onnxruntime as ort

    from config import get_settings

    dossier = get_settings().models_dir / "texte"
    dossier.mkdir(parents=True, exist_ok=True)
    modele = dossier / "ppocrv4_det.onnx"
    if not modele.is_file():
        logger.info("Téléchargement du détecteur de texte PP-OCRv4…")
        partiel = dossier / "ppocrv4_det.part"
        with httpx.stream("GET", URL_DETECTEUR, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with partiel.open("wb") as f:
                for bloc in r.iter_bytes():
                    f.write(bloc)
        partiel.replace(modele)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(modele), options, providers=["CPUExecutionProvider"])


def carte_texte(roi):
    """Probabilité « texte » de chaque pixel d'une image BGR (même taille que `roi`, de 0 à 1)."""
    import cv2
    import numpy as np

    h, w = roi.shape[:2]
    echelle = min(1.0, 640 / max(h, w))  # le texte reste lisible pour le détecteur, en 2 à 3 fois moins de calcul
    hh, ww = max(32, round(h * echelle / 32) * 32), max(32, round(w * echelle / 32) * 32)
    x = cv2.resize(roi, (ww, hh)).astype(np.float32) / 255.0
    x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    session = _detecteur()
    carte = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
    return cv2.resize(carte[0, 0], (w, h))


def effacer_texte(image, zone: tuple[int, int, int, int], memoire: dict | None = None) -> None:
    """Efface, dans `image` (BGR, modifiée sur place), le texte incrusté situé dans la zone.

    Le texte est repéré par un détecteur de texte (PP-OCR) : seules les lettres, leur contour et leur
    ombre sont reconstruits à partir des pixels voisins (inpainting), le reste de la zone n'est pas
    touché et une image sans texte n'est pas modifiée. Ni flou ni cadre.

    `memoire` (un dict propre à la vidéo) évite de tout recalculer quand la zone n'a pas changé depuis
    l'image précédente (plan fixe, sous-titre identique) : le résultat précédent est réutilisé."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = zone
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return
    if memoire is not None:
        avant = memoire.get("source")
        if avant is not None and avant.shape == roi.shape and cv2.absdiff(avant, roi).mean() < 1.0:
            roi[:] = memoire["resultat"]
            return
        memoire["source"] = roi.copy()
        memoire["resultat"] = roi  # remplacé ci-dessous si du texte est effacé
    noyau = (carte_texte(roi) > 0.3).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(noyau, connectivity=8)
    if n <= 1:
        return
    # Le détecteur rend le « cœur » rétréci de chaque ligne de texte : chaque ligne est élargie comme dans
    # DB (« unclip » : décalage = surface × 1,8 / périmètre), pour couvrir accents, jambages et contour
    lignes = np.zeros(noyau.shape, np.uint8)
    for x, y, l, h, surface in stats[1:]:
        if surface < 6:
            continue
        d = int(np.ceil(surface * 1.8 / (2 * (l + h)))) + 2
        cote = d + int(1.5 * (h + 2 * d))  # émoji ou ponctuation en bout de ligne, que le détecteur ignore
        cv2.rectangle(lignes, (x - cote, y - d), (x + l + cote, y + h + d), 1, -1)
    if not lignes.any():
        return
    # Dans ces lignes, seuls les pixels des lettres (clairs ou jaunes), de leur contour et de leur ombre
    # (sombres) sont reconstruits : le décor de teinte moyenne (ciel, peau…) reste intact
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    teinte, saturation, valeur = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lettres = (((valeur >= 165) & (saturation <= 90)) | (valeur <= 95)
               | ((teinte >= 18) & (teinte <= 38) & (saturation >= 90) & (valeur >= 160))
               | ((saturation >= 150) & (valeur >= 120))  # émojis et texte de couleur vive
               | (((teinte <= 8) | (teinte >= 160)) & (saturation >= 50) & (valeur >= 120)))  # cœurs rouges/roses
    masque = cv2.dilate((lettres & (lignes > 0)).astype(np.uint8) * 255,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    masque[lignes == 0] = 0
    # Reconstruction limitée au rectangle des lignes (plus une marge de pixels voisins) : bien plus rapide
    ys, xs = np.nonzero(masque)
    if ys.size:
        bord = 8
        a, b = max(0, ys.min() - bord), min(roi.shape[0], ys.max() + bord + 1)
        g, d = max(0, xs.min() - bord), min(roi.shape[1], xs.max() + bord + 1)
        roi[a:b, g:d] = cv2.inpaint(np.ascontiguousarray(roi[a:b, g:d]), masque[a:b, g:d], 3, cv2.INPAINT_TELEA)
    if memoire is not None:
        memoire["resultat"] = roi.copy()


# ---------------------------------------------------------------------------------------------------
# Filtres FFmpeg
# ---------------------------------------------------------------------------------------------------

def _filtres_qualite(r: dict[str, Any]) -> list[str]:
    if not r["qualite"]:
        return []
    # Débruitage léger (sans effet « peinture ») puis netteté modérée de la luminance ; taille inchangée
    return ["hqdn3d=1.5:1.5:6:6", "unsharp=5:5:0.5:5:5:0"]


def graphe(r: dict[str, Any], ass: str | None, entree: str = "0:v") -> str:
    """Graphe de filtres : débruitage → netteté → sous-titres et logo. La taille de l'image ne change
    jamais. `ass` est un nom de fichier relatif au dossier de travail de FFmpeg (évite les « C: »)."""
    chaine = _filtres_qualite(r) + ([f"ass={ass}"] if ass else [])
    return f"[{entree}]{','.join(chaine) or 'null'}[v]"


def _arguments_metadonnees(r: dict[str, Any], codec: str | None, entree: int = 0) -> list[str]:
    """Sans métadonnées : ni titre, date, lieu, appareil, logiciel, chapitres, ni données cachées dans le flux."""
    if not r["sans_metadonnees"]:
        return ["-map_metadata", str(entree)]
    arguments = ["-map_metadata", "-1", "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1", "-map_chapters", "-1",
                 "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact"]
    if codec == "h264":
        arguments += ["-bsf:v", "filter_units=remove_types=6"]  # SEI : réglages de l'encodeur, appareil…
    elif codec == "hevc":
        arguments += ["-bsf:v", "filter_units=remove_types=39|40"]
    return arguments


def _instant_apercu(segments: list[dict[str, Any]], duree: float) -> float:
    """Instant par défaut de l'aperçu : au milieu de la première phrase (des sous-titres y sont visibles)."""
    if segments:
        seg = segments[0]
        return float(seg["debut"]) + 0.45 * float(seg.get("duree_doublage") or (seg["fin"] - seg["debut"]))
    return min(3.0, duree / 3)


def _executer(commande: list[str], message: str, entree: bytes | None = None, cwd: Path | None = None) -> bytes:
    resultat = subprocess.run(commande, input=entree, capture_output=True, cwd=cwd)
    if resultat.returncode != 0 or not resultat.stdout:
        raise FFmpegError(f"{message} : {resultat.stderr.decode(errors='replace').strip()[-300:]}")
    return resultat.stdout


def apercu(source: Path, segments: list[dict[str, Any]], reglages: dict[str, Any], dossier: Path,
           instant: float | None, largeur_max: int = 960) -> bytes:
    """Une image JPEG de la vidéo habillée à l'instant voulu, pour régler l'habillage avant l'export."""
    import numpy as np

    r = normaliser(reglages)
    duree = probe_duration(source)
    t = min(max(0.0, instant if instant is not None else _instant_apercu(segments, duree)), max(0.0, duree - 0.05))
    taille = taille_affichee(source)
    ffmpeg = _require("ffmpeg")
    nom = f"apercu_{uuid.uuid4().hex[:8]}.ass"
    ass = None
    if r["sous_titres"] or r["logo"]:
        ecrire_ass(segments, r, taille, duree, dossier / nom)
        ass = nom
    reduire = f"scale='min({largeur_max},iw)':-2"
    try:
        if r["flou"]:
            # Image brute à cet instant, lettres effacées, puis sous-titres et logo (horodatage remis à `t`)
            filtres = ",".join(_filtres_qualite(r)) or "null"
            brute = _executer([ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(source.resolve()), "-vf", filtres,
                               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], "Aperçu impossible")
            image = np.frombuffer(brute, np.uint8)[: taille.hauteur * taille.largeur * 3].copy()
            image = image.reshape(taille.hauteur, taille.largeur, 3)
            effacer_texte(image, zone_pixels(r["zone"], taille))
            chaine = [f"setpts=PTS+{t:.3f}/TB"] + ([f"ass={ass}"] if ass else []) + [reduire]
            return _executer([ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{taille.largeur}x{taille.hauteur}", "-i", "pipe:0", "-vf", ",".join(chaine),
                              "-frames:v", "1", "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
                             "Aperçu impossible", entree=image.tobytes(), cwd=dossier)
        # -copyts : l'image garde son horodatage réel, les sous-titres de cet instant s'affichent
        return _executer([ffmpeg, "-v", "error", "-copyts", "-ss", f"{t:.3f}", "-i", str(source.resolve()),
                          "-filter_complex", graphe(r, ass) + f";[v]{reduire}[a]", "-map", "[a]", "-frames:v", "1",
                          "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "pipe:1"], "Aperçu impossible", cwd=dossier)
    finally:
        (dossier / nom).unlink(missing_ok=True)


def _attendre(processus: subprocess.Popen, sortie: Path, journal: Path) -> None:
    if processus.wait() != 0:
        sortie.unlink(missing_ok=True)
        message = journal.read_text(encoding="utf-8", errors="replace").strip()[-300:]
        raise FFmpegError(f"L'habillage de la vidéo a échoué : {message}")


def habiller(source: Path, segments: list[dict[str, Any]], reglages: dict[str, Any], sortie: Path,
             on_progress: Callable[[float], None] | None = None,
             doit_arreter: Callable[[], bool] | None = None) -> Path:
    """Produit la vidéo habillée `sortie` : même conteneur, codec, résolution et cadence que `source`.

    Le son est copié tel quel. Sans élément visuel (seulement la suppression des métadonnées), l'image
    est copiée elle aussi, sans perte. Pour effacer d'anciens sous-titres, les images passent une à une
    par `effacer_texte` (un décodeur FFmpeg → Python → un encodeur FFmpeg : mémoire bornée quelle que
    soit la durée). `doit_arreter` est consulté pendant le rendu : s'il renvoie vrai, FFmpeg est arrêté
    et `InterruptedError` levée."""
    r = normaliser(reglages)
    fmt = sonder(source)
    duree = probe_duration(source)
    dossier = sortie.parent
    ffmpeg = _require("ffmpeg")
    ass = dossier / "habillage.ass"
    journal = dossier / "habillage.log"
    taille = taille_affichee(source) if _visuel(r) else None
    avec_ass = bool(taille) and (r["sous_titres"] or r["logo"])
    if avec_ass:
        ecrire_ass(segments, r, taille, duree, ass)
    codec_sortie = ({"libx264": "h264", "libx265": "hevc"}.get(arguments_video(fmt)[1]) if taille
                    else fmt.codec_video)

    def interrompre(*processus: subprocess.Popen) -> None:
        for p in processus:
            p.kill()
            p.wait()
        sortie.unlink(missing_ok=True)
        raise InterruptedError("Habillage interrompu.")

    with journal.open("w", encoding="utf-8", errors="replace") as erreurs:
        if taille and r["flou"]:
            import numpy as np

            cadence = fmt.cadence or "25"
            decodeur = subprocess.Popen(
                [ffmpeg, "-v", "error", "-i", str(source.resolve()), "-map", "0:v:0",
                 "-vf", ",".join(_filtres_qualite(r)) or "null", "-fps_mode", "cfr", "-r", cadence,
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                stdout=subprocess.PIPE, stderr=erreurs)
            encodeur = subprocess.Popen(
                [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{taille.largeur}x{taille.hauteur}", "-framerate", cadence, "-i", "pipe:0",
                 "-i", str(source.resolve()), "-filter_complex", graphe(r, ass.name if avec_ass else None),
                 "-map", "[v]", "-map", "1:a?", *arguments_video(fmt), "-c:a", "copy",
                 *_arguments_metadonnees(r, codec_sortie, entree=1), *_conteneur(fmt), str(sortie.resolve())],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=erreurs, cwd=dossier)
            zone = zone_pixels(r["zone"], taille)
            memoire: dict = {}
            octets = taille.largeur * taille.hauteur * 3
            num, _, den = cadence.partition("/")
            total = max(1.0, duree * float(num) / float(den or 1))
            n = 0
            try:
                while True:
                    brute = decodeur.stdout.read(octets)  # type: ignore[union-attr]
                    if len(brute) < octets:
                        break
                    if doit_arreter and doit_arreter():
                        interrompre(decodeur, encodeur)
                    image = np.frombuffer(brute, np.uint8).reshape(taille.hauteur, taille.largeur, 3).copy()
                    effacer_texte(image, zone, memoire)
                    encodeur.stdin.write(image.tobytes())  # type: ignore[union-attr]
                    n += 1
                    if on_progress and n % 25 == 0:
                        on_progress(min(0.99, n / total))
            except BrokenPipeError:
                pass  # l'encodeur s'est arrêté : son erreur est dans le journal
            finally:
                if encodeur.stdin and not encodeur.stdin.closed:
                    try:
                        encodeur.stdin.close()
                    except OSError:
                        pass  # encodeur déjà arrêté
                decodeur.stdout.close()  # type: ignore[union-attr]
                decodeur.wait()
            _attendre(encodeur, sortie, journal)
        else:
            commande = [ffmpeg, "-y", "-v", "error", "-nostats", "-progress", "pipe:1", "-i", str(source.resolve())]
            if taille:
                commande += ["-filter_complex", graphe(r, ass.name if avec_ass else None),
                             "-map", "[v]", "-map", "0:a?", *arguments_video(fmt), "-c:a", "copy"]
            else:
                commande += ["-map", "0:v:0", "-map", "0:a?", "-c", "copy"]
            commande += [*_arguments_metadonnees(r, codec_sortie), *_conteneur(fmt), str(sortie.resolve())]
            processus = subprocess.Popen(commande, stdout=subprocess.PIPE, stderr=erreurs, text=True, cwd=dossier)
            try:
                for ligne in processus.stdout:  # type: ignore[union-attr]
                    if doit_arreter and doit_arreter():
                        interrompre(processus)
                    if ligne.startswith("out_time_us=") and on_progress and duree > 0:
                        try:
                            on_progress(min(0.99, int(ligne.split("=", 1)[1]) / 1e6 / duree))
                        except ValueError:
                            pass  # « N/A » au tout début
            finally:
                if processus.poll() is None:
                    processus.wait()
            _attendre(processus, sortie, journal)
    ass.unlink(missing_ok=True)
    journal.unlink(missing_ok=True)
    if on_progress:
        on_progress(1.0)
    return sortie
