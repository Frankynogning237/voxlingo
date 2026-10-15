"""Lecture des scènes : quels personnages sont à l'image, et quelle bouche bouge.

Complète l'écoute (pipeline/diarize.py) : deux personnages doublés par la même voix (vidéos
générées par IA) ont des visages différents, et une même personne dont la voix change de ton
garde le même visage. Tout est local et léger (CPU) :

- **MediaPipe Face Landmarker** (Google, ~4 Mo) : visages et 478 repères, dont l'ouverture de
  la mâchoire (`jawOpen`) qui révèle qui parle ;
- **ArcFace MobileFaceNet** (InsightFace `w600k_mbf`, ~13 Mo, usage non commercial) : empreinte
  du visage, complétée par les couleurs des cheveux et de la tenue, pour reconnaître un même
  personnage d'un plan à l'autre.

La vidéo est lue à 8 images/s en 480 px de large, en flux (mémoire bornée, toute durée).
Chaque visage est suivi à l'intérieur d'un plan (une « piste ») ; les pistes sont ensuite
regroupées en personnages. Le résultat est gardé dans `vision.json` du projet.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import get_settings

logger = logging.getLogger("voxlingo.vision")

IMAGES_PAR_S = 8
LARGEUR = 480
TAILLE_MIN_VISAGE = 30  # px (front → menton) : en dessous, visage trop petit pour lire les lèvres
SEUIL_PLAN = 30.0  # écart moyen entre deux images réduites au-delà duquel la caméra a changé de plan
SEUIL_PERSONNAGE = 0.45  # distance (apparence + visage + couleurs) en dessous de laquelle deux pistes sont la même personne
MOUVEMENT_MIN = 0.004  # mouvement moyen de la mâchoire par image pour « parler »
AVANCE_MIN = 2.0  # la bouche qui parle doit bouger au moins 2 fois plus que les autres
MODELE_APPARENCE = "facebook/dinov2-small"
POIDS_APPARENCE, POIDS_VISAGE, POIDS_COULEURS = 0.5, 0.25, 0.25
URL_REPERES = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
               "face_landmarker.task")
URL_ARCFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
GABARIT_ARCFACE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _dossier_modeles() -> Path:
    dossier = get_settings().models_dir / "vision"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier


def _modeles() -> tuple[Path, Path]:
    """Chemins des deux modèles, téléchargés au premier usage depuis leurs sources officielles."""
    import zipfile

    import httpx

    dossier = _dossier_modeles()
    reperes, arcface = dossier / "face_landmarker.task", dossier / "w600k_mbf.onnx"
    if not reperes.is_file():
        logger.info("Téléchargement de MediaPipe Face Landmarker…")
        with httpx.stream("GET", URL_REPERES, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with (dossier / "face_landmarker.part").open("wb") as f:
                for bloc in r.iter_bytes():
                    f.write(bloc)
        (dossier / "face_landmarker.part").replace(reperes)
    if not arcface.is_file():
        logger.info("Téléchargement d'ArcFace (InsightFace buffalo_s)…")
        archive = dossier / "buffalo_s.zip"
        with httpx.stream("GET", URL_ARCFACE, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for bloc in r.iter_bytes():
                    f.write(bloc)
        with zipfile.ZipFile(archive) as z:
            z.extract("w600k_mbf.onnx", dossier)
        archive.unlink()
    return reperes, arcface


def _histogramme(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    """Histogramme de couleurs (HSV) d'une zone de l'image, normalisé."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    histo = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return histo / (histo.sum() or 1.0)


def _lire_pistes(video: Path, on_progress: Callable[[float], None] | None, duree_s: float | None) -> list[dict]:
    """Lit la vidéo et renvoie les pistes de visages (un visage suivi dans un plan)."""
    import cv2
    import mediapipe as mp
    import onnxruntime as ort

    from pipeline.extract import _require

    chemin_reperes, chemin_arcface = _modeles()
    sonde = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip()
    w0, h0 = (int(v) for v in sonde.split(",")[:2])
    hauteur = max(2, int(round(h0 * LARGEUR / w0 / 2) * 2))
    flux = subprocess.Popen(
        [_require("ffmpeg"), "-v", "error", "-i", str(video), "-vf", f"fps={IMAGES_PAR_S},scale={LARGEUR}:{hauteur}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(chemin_reperes)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=4, output_face_blendshapes=True,
        min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
    arcface = ort.InferenceSession(str(chemin_arcface), providers=["CPUExecutionProvider"])
    entree = arcface.get_inputs()[0].name
    octets = LARGEUR * hauteur * 3

    pistes: list[dict] = []
    actives: list[dict] = []
    plan, precedente, n = 0, None, 0
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as reperes:
            while len(brut := flux.stdout.read(octets)) == octets:  # type: ignore[union-attr]
                image = np.frombuffer(brut, np.uint8).reshape(hauteur, LARGEUR, 3)
                t = n / IMAGES_PAR_S
                reduite = cv2.resize(image, (32, 32)).astype(np.float32)
                if precedente is not None and np.abs(reduite - precedente).mean() > SEUIL_PLAN:
                    plan += 1
                    actives = []
                precedente = reduite
                resultat = reperes.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=image),
                                                    int(t * 1000))
                prises: list[dict] = []
                for k, points in enumerate(resultat.face_landmarks):
                    P = np.array([[p.x * LARGEUR, p.y * hauteur] for p in points], dtype=np.float32)
                    taille = float(np.linalg.norm(P[10] - P[152]))  # front → menton
                    if taille < TAILLE_MIN_VISAGE:
                        continue
                    cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
                    machoire = next((b.score for b in resultat.face_blendshapes[k] if b.category_name == "jawOpen"), 0.0)
                    cinq = np.array([(P[33] + P[133]) / 2, (P[362] + P[263]) / 2, P[1], P[61], P[291]], dtype=np.float32)
                    M, _ = cv2.estimateAffinePartial2D(cinq, GABARIT_ARCFACE)
                    alignee = cv2.warpAffine(image, M, (112, 112)).astype(np.float32)
                    e = arcface.run(None, {entree: ((alignee - 127.5) / 127.5).transpose(2, 0, 1)[None]})[0][0]
                    largeur = float(P[:, 0].max() - P[:, 0].min())
                    detection = {
                        "n": n, "cx": cx, "cy": cy, "taille": taille, "machoire": round(float(machoire), 4),
                        "e": e / (np.linalg.norm(e) or 1.0),
                        "cheveux": _histogramme(image, cx - largeur * 0.7, P[10][1] - taille * 0.6,
                                                cx + largeur * 0.7, P[10][1] + taille * 0.1),
                        "tenue": _histogramme(image, cx - largeur, P[152][1] + taille * 0.15,
                                              cx + largeur, P[152][1] + taille * 1.2),
                    }
                    # Le personnage (tête, coiffure et haut de la tenue, sans les voisins), pour l'empreinte d'apparence DINOv2
                    portrait = _decouper(image, cx - largeur * 0.8, P[10][1] - taille * 0.35,
                                         cx + largeur * 0.8, P[152][1] + taille * 0.9)
                    # Suivi dans le plan : la piste active la plus proche
                    meilleure = None
                    for piste in actives:
                        derniere = piste["d"][-1]
                        ecart = np.hypot(derniere["cx"] - cx, derniere["cy"] - cy) / taille
                        if piste not in prises and n - derniere["n"] <= IMAGES_PAR_S and ecart < 0.6 and (
                                meilleure is None or ecart < meilleure[0]):
                            meilleure = (ecart, piste)
                    if meilleure:
                        piste = meilleure[1]
                        piste["d"].append(detection)
                    else:
                        piste = {"plan": plan, "d": [detection], "portraits": []}
                        pistes.append(piste)
                    # Quelques portraits par piste (début, puis tous les ~2 s) suffisent
                    if portrait is not None and (len(piste["d"]) - 1) % (2 * IMAGES_PAR_S) == 0 \
                            and len(piste["portraits"]) < 4:
                        piste["portraits"].append(portrait)
                    prises.append(piste)
                actives = [p for p in pistes if p["plan"] == plan and p["d"][-1]["n"] >= n - IMAGES_PAR_S]
                n += 1
                if on_progress and duree_s and n % IMAGES_PAR_S == 0:
                    on_progress(min(0.95, n / IMAGES_PAR_S / duree_s))
    finally:
        flux.kill()
    logger.info("Lecture des scènes : %d images, %d plans, %d pistes de visages", n, plan + 1, len(pistes))
    return [p for p in pistes if len(p["d"]) >= 3]


def _decouper(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    """Zone de l'image (bornée), mise à 224 × 224 pour DINOv2."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.resize(np.ascontiguousarray(image[y0:y1, x0:x1]), (224, 224), interpolation=cv2.INTER_AREA)


def _apparences(pistes: list[dict]) -> np.ndarray:
    """Empreinte d'apparence (DINOv2 small, ~90 Mo) de chaque piste : moyenne de ses portraits."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = _dossier_modeles() / "hf"
    processeur = AutoImageProcessor.from_pretrained(MODELE_APPARENCE, cache_dir=cache)
    modele = AutoModel.from_pretrained(MODELE_APPARENCE, cache_dir=cache).eval()
    portraits = [(i, im) for i, p in enumerate(pistes) for im in p.get("portraits", [])]
    sommes = np.zeros((len(pistes), modele.config.hidden_size), dtype=np.float32)
    try:
        for debut in range(0, len(portraits), 16):
            lot = portraits[debut:debut + 16]
            with torch.no_grad():
                sortie = modele(**processeur(images=[im for _, im in lot], return_tensors="pt"))
            vecteurs = sortie.pooler_output.numpy()
            for (i, _), v in zip(lot, vecteurs, strict=True):
                sommes[i] += v / (np.linalg.norm(v) or 1.0)
    finally:
        del modele
    return sommes / np.maximum(np.linalg.norm(sommes, axis=1, keepdims=True), 1e-9)


def _moyenne(valeurs: list[np.ndarray | None], normaliser: bool = False) -> np.ndarray | None:
    presentes = [v for v in valeurs if v is not None]
    if not presentes:
        return None
    v = np.mean(presentes, axis=0)
    return v / (np.linalg.norm(v) or 1.0) if normaliser else v


def _personnages(pistes: list[dict]) -> list[int]:
    """Regroupe les pistes en personnages : apparence générale (DINOv2), visage (ArcFace),
    couleurs des cheveux et de la tenue."""
    from sklearn.cluster import AgglomerativeClustering

    m = len(pistes)
    if m <= 1:
        return [0] * m
    E = np.stack([_moyenne([d["e"] for d in p["d"]], normaliser=True) for p in pistes])

    def bhattacharyya(cle: str) -> np.ndarray:
        histos = [_moyenne([d[cle] for d in p["d"]]) for p in pistes]
        connus = np.array([h is not None for h in histos])
        racines = np.sqrt(np.stack([h if h is not None else np.zeros(192) for h in histos]))
        distance = np.sqrt(np.clip(1 - racines @ racines.T, 0, None))
        distance[~(connus[:, None] & connus[None, :])] = 0.5  # couleur inconnue : neutre
        return distance

    A = _apparences(pistes)
    sans_portrait = ~A.any(axis=1)
    distance_apparence = 1 - A @ A.T
    distance_apparence[sans_portrait, :] = distance_apparence[:, sans_portrait] = 0.5
    D = (POIDS_APPARENCE * distance_apparence + POIDS_VISAGE * (1 - E @ E.T)
         + POIDS_COULEURS / 2 * (bhattacharyya("cheveux") + bhattacharyya("tenue")))
    # Deux visages à l'image en même temps sont deux personnes différentes
    for i in range(m):
        for j in range(i + 1, m):
            a, b = pistes[i], pistes[j]
            if a["plan"] == b["plan"] and a["d"][0]["n"] <= b["d"][-1]["n"] and b["d"][0]["n"] <= a["d"][-1]["n"]:
                D[i, j] = D[j, i] = 2.0
    np.fill_diagonal(D, 0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=SEUIL_PERSONNAGE).fit_predict(D).tolist()


def analyser(video: Path, dossier_projet: Path, duree_s: float | None = None,
             on_progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Personnages et mouvements de bouche de la vidéo (mis en cache dans `vision.json`).

    Renvoie {"ips": 8, "pistes": [{"perso": 0, "images": [n…], "machoire": [0.12…]}…], "personnages": N}.
    """
    cache = dossier_projet / "vision.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    pistes = _lire_pistes(video, on_progress, duree_s)
    persos = _personnages(pistes)
    resultat = {
        "ips": IMAGES_PAR_S,
        "personnages": len(set(persos)),
        "pistes": [{"perso": int(g), "images": [d["n"] for d in p["d"]], "machoire": [d["machoire"] for d in p["d"]]}
                   for p, g in zip(pistes, persos, strict=True)],
    }
    _vignettes(video, dossier_projet, pistes, persos)
    cache.write_text(json.dumps(resultat), encoding="utf-8")
    logger.info("Lecture des scènes : %d personnage(s) à l'image", resultat["personnages"])
    if on_progress:
        on_progress(1.0)
    return resultat


def _vignettes(video: Path, dossier_projet: Path, pistes: list[dict], persos: list[int]) -> None:
    """Enregistre un visage de chaque personnage (visages/perso_N.jpg) : le plus grand de sa plus
    longue piste, la plus représentative de lui."""
    from pipeline.extract import _require

    dossier = dossier_projet / "visages"
    dossier.mkdir(exist_ok=True)
    for g in set(persos):
        piste = max((p for p, gp in zip(pistes, persos, strict=True) if gp == g), key=lambda p: len(p["d"]))
        detection = max(piste["d"], key=lambda d: d["taille"])
        cote = int(detection["taille"] * 1.4)
        x, y = int(max(0.0, detection["cx"] - cote / 2)), int(max(0.0, detection["cy"] - cote / 2))
        # Recadrage carré autour du visage, borné aux dimensions de l'image (vidéos verticales)
        recadrage = (f"crop='min({cote},min(iw,ih))':'min({cote},min(iw,ih))':"
                     f"'min({x},iw-ow)':'min({y},ih-oh)'")
        subprocess.run(
            [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{detection['n'] / IMAGES_PAR_S:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={LARGEUR}:-2,{recadrage},scale=160:160", "-q:v", "4",
             str(dossier / f"perso_{g}.jpg")], capture_output=True, check=False)


def qui_parle(vision: dict[str, Any], debut: float, fin: float) -> tuple[int | None, dict[int, float]]:
    """Personnage dont la bouche bouge pendant [debut, fin], ou None si l'image ne permet pas de
    trancher (personne à l'image, voix hors champ, plusieurs bouches qui bougent autant)."""
    ips = vision["ips"]
    a, b = int(debut * ips), int(np.ceil(fin * ips))
    nb_images = max(1, b - a)
    mouvement: dict[int, float] = {}
    for piste in vision["pistes"]:
        valeurs = [m for n, m in zip(piste["images"], piste["machoire"], strict=True) if a <= n <= b]
        if len(valeurs) >= 2:
            mouvement[piste["perso"]] = mouvement.get(piste["perso"], 0.0) + float(np.abs(np.diff(valeurs)).sum())
    scores = {p: v / nb_images for p, v in mouvement.items()}
    if not scores:
        return None, scores
    tri = sorted(scores.items(), key=lambda kv: -kv[1])
    premier, score = tri[0]
    second = tri[1][1] if len(tri) > 1 else 0.0
    if score >= MOUVEMENT_MIN and score >= AVANCE_MIN * second:
        return premier, scores
    return None, scores


def visibles(vision: dict[str, Any], debut: float, fin: float, marge: float = 0.0) -> dict[int, int]:
    """Personnages à l'image pendant [debut - marge, fin + marge], avec leur nombre d'images."""
    ips = vision["ips"]
    a, b = int((debut - marge) * ips), int(np.ceil((fin + marge) * ips))
    presence: dict[int, int] = {}
    for piste in vision["pistes"]:
        nombre = sum(1 for n in piste["images"] if a <= n <= b)
        if nombre:
            presence[piste["perso"]] = presence.get(piste["perso"], 0) + nombre
    return presence
