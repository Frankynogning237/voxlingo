"""Bibliothèque de voix : 20 voix de tous âges et les voix personnelles de l'utilisateur.

Chaque voix porte un sexe, un âge (enfant, jeune, adulte, senior) et un registre (grave,
médium, aigu) mesuré sur son aperçu (fréquence fondamentale médiane). L'âge se corrige
dans l'interface après écoute.

Une voix personnelle est un échantillon de 6 à 15 s (enfant, personne âgée, voix
particulière…) : XTTS la clone comme la voix d'origine d'une vidéo.
Le tout est enregistré dans data/voix/banque.json, les fichiers audio à côté.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np

from config import get_settings

logger = logging.getLogger("voxlingo.voice_bank")

Sexe = Literal["femme", "homme"]
Age = Literal["enfant", "jeune", "adulte", "senior"]

# 20 voix couvrant tous les âges. Toutes partent d'une voix studio fournie avec XTTS v2
# (speakers_xtts.pth, uniquement des voix d'adultes), dont le sexe a été vérifié par la
# mesure de la hauteur de voix (hommes : 88 à 123 Hz ; femmes : 168 à 222 Hz). Les voix d'enfants, de jeunes et de
# seniors sont obtenues en transformant ces voix après synthèse (FFmpeg Rubber Band) :
#   enfant : hauteur ×1,25 à ×1,45, résonances relevées ;
#   jeune  : légèrement plus haute et plus rapide ;
#   senior : plus grave, plus lente, léger tremblement.
# Plus crédible qu'une voix d'adulte, moins naturel qu'un vrai enregistrement : pour un
# rendu parfait, l'utilisateur peut ajouter une voix personnelle enregistrée.
VOIX_BIBLIOTHEQUE: list[dict[str, Any]] = [
    # Enfants
    {"id": "lila", "nom": "Lila", "sexe": "femme", "age": "enfant", "base": "Daisy Studious", "pitch": 1.45},
    {"id": "zoe", "nom": "Zoé", "sexe": "femme", "age": "enfant", "base": "Gracie Wise", "pitch": 1.4},
    {"id": "theo", "nom": "Théo", "sexe": "homme", "age": "enfant", "base": "Ana Florence", "pitch": 1.3},
    {"id": "leo", "nom": "Léo", "sexe": "homme", "age": "enfant", "base": "Sofia Hellen", "pitch": 1.25},
    # Jeunes
    {"id": "ines", "nom": "Inès", "sexe": "femme", "age": "jeune", "base": "Tammie Ema", "pitch": 1.06, "tempo": 1.04},
    {"id": "chloe", "nom": "Chloé", "sexe": "femme", "age": "jeune", "base": "Alison Dietlinde", "pitch": 1.06, "tempo": 1.04},
    {"id": "hugo", "nom": "Hugo", "sexe": "homme", "age": "jeune", "base": "Zacharie Aimilios", "pitch": 1.06, "tempo": 1.04},
    {"id": "nathan", "nom": "Nathan", "sexe": "homme", "age": "jeune", "base": "Royston Min", "pitch": 1.06, "tempo": 1.04},
    # Adultes
    {"id": "claire", "nom": "Claire", "sexe": "femme", "age": "adulte", "base": "Claribel Dervla"},
    {"id": "sophie", "nom": "Sophie", "sexe": "femme", "age": "adulte", "base": "Annmarie Nele"},
    {"id": "nadia", "nom": "Nadia", "sexe": "femme", "age": "adulte", "base": "Asya Anara"},
    {"id": "brenda", "nom": "Brenda", "sexe": "femme", "age": "adulte", "base": "Brenda Stern"},
    {"id": "marc", "nom": "Marc", "sexe": "homme", "age": "adulte", "base": "Damien Black"},
    {"id": "karim", "nom": "Karim", "sexe": "homme", "age": "adulte", "base": "Kumar Dahl"},
    {"id": "julien", "nom": "Julien", "sexe": "homme", "age": "adulte", "base": "Luis Moray"},
    {"id": "victor", "nom": "Victor", "sexe": "homme", "age": "adulte", "base": "Torcull Diarmuid"},
    # Seniors
    {"id": "jeanne", "nom": "Mamie Jeanne", "sexe": "femme", "age": "senior", "base": "Henriette Usha",
     "pitch": 0.94, "tempo": 0.92, "vibrato": 0.12},
    {"id": "rose", "nom": "Mamie Rose", "sexe": "femme", "age": "senior", "base": "Tanja Adelina",
     "pitch": 0.94, "tempo": 0.92, "vibrato": 0.12},
    {"id": "georges", "nom": "Papi Georges", "sexe": "homme", "age": "senior", "base": "Abrahan Mack",
     "pitch": 0.92, "tempo": 0.9, "vibrato": 0.14},
    {"id": "albert", "nom": "Papi Albert", "sexe": "homme", "age": "senior", "base": "Gilberto Mathias",
     "pitch": 0.92, "tempo": 0.9, "vibrato": 0.14},
]
VERSION_BIBLIOTHEQUE = 2

TEXTE_APERCU = "Bonjour, voici un aperçu de ma voix pour vos doublages. J'espère qu'elle vous plaira !"
SEUIL_SEXE_HZ = 165  # en dessous : voix plutôt masculine
SEUIL_ENFANT_HZ = 270  # au-dessus : voix d'enfant probable
PENALITE_AGE_HZ = 25  # écart ajouté aux voix jeunes et seniors lors de l'attribution automatique
_verrou = threading.Lock()


def dossier_voix() -> Path:
    dossier = get_settings().data_dir / "voix"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier


def _fichier_banque() -> Path:
    return dossier_voix() / "banque.json"


def _initiale() -> list[dict[str, Any]]:
    return [
        {
            "id": v["id"], "nom": v["nom"], "source": "xtts", "xtts": v["base"],
            "transformation": {k: v[k] for k in ("pitch", "tempo", "vibrato") if k in v},
            "sexe": v["sexe"], "age": v["age"], "registre": None, "f0_hz": None,
            "version": VERSION_BIBLIOTHEQUE,
        }
        for v in VOIX_BIBLIOTHEQUE
    ]


def lire_banque() -> list[dict[str, Any]]:
    with _verrou:
        chemin = _fichier_banque()
        banque = json.loads(chemin.read_text(encoding="utf-8")) if chemin.is_file() else []
        if not any(v.get("version") == VERSION_BIBLIOTHEQUE for v in banque):
            # Nouvelle version de la bibliothèque : on garde les voix personnelles, on remplace les autres
            for ancienne in banque:
                if ancienne["source"] == "xtts":
                    fichier_apercu(ancienne["id"]).unlink(missing_ok=True)
            banque = _initiale() + [v for v in banque if v["source"] == "perso"]
            chemin.write_text(json.dumps(banque, ensure_ascii=False, indent=1), encoding="utf-8")
    for voix in banque:
        voix["apercu"] = fichier_apercu(voix["id"]).is_file()
    return banque


def ecrire_banque(banque: list[dict[str, Any]]) -> None:
    with _verrou:
        propre = [{k: v for k, v in voix.items() if k != "apercu"} for voix in banque]
        _fichier_banque().write_text(json.dumps(propre, ensure_ascii=False, indent=1), encoding="utf-8")


def trouver(voix_id: str) -> dict[str, Any] | None:
    return next((v for v in lire_banque() if v["id"] == voix_id), None)


def fichier_apercu(voix_id: str) -> Path:
    return dossier_voix() / "apercus" / f"{voix_id}.wav"


def spec(voix: dict[str, Any]) -> str:
    """Voix au format attendu par pipeline/voice.py : « xtts:<nom> » ou « ref:<échantillon> »,
    suivie de sa transformation d'âge éventuelle, par exemple « xtts:Daisy Studious|pitch=1.45 »."""
    base = f"xtts:{voix['xtts']}" if voix["source"] == "xtts" else f"ref:{dossier_voix() / voix['fichier']}"
    transformation = voix.get("transformation") or {}
    return "|".join([base, *(f"{k}={v}" for k, v in sorted(transformation.items()))])


def separer_spec(voix: str) -> tuple[str, dict[str, float]]:
    """« xtts:Daisy Studious|pitch=1.45 » → (« xtts:Daisy Studious », {"pitch": 1.45})."""
    base, *parametres = voix.split("|")
    return base, {k: float(v) for k, _, v in (p.partition("=") for p in parametres)}


def transformer(fichier: Path, parametres: dict[str, float]) -> None:
    """Applique la transformation d'âge (hauteur, résonances, tempo, tremblement) à un WAV, sur place."""
    import subprocess

    from pipeline.extract import FFmpegError, _require

    if not parametres:
        return
    filtres = []
    pitch, tempo = parametres.get("pitch", 1.0), parametres.get("tempo", 1.0)
    if pitch != 1.0 or tempo != 1.0:
        # formant=shifted : les résonances suivent la hauteur (timbre plus « petit » pour un enfant)
        filtres.append(f"rubberband=pitch={pitch}:tempo={tempo}:formant=shifted:pitchq=quality")
    if parametres.get("vibrato"):
        filtres.append(f"vibrato=f=5.5:d={parametres['vibrato']}")
    sortie = fichier.with_name(fichier.stem + "_age.wav")
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(fichier), "-af", ",".join(filtres),
         "-ar", "24000", "-ac", "1", str(sortie)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError(f"Transformation de la voix impossible : {resultat.stderr.strip()[:200]}")
    sortie.replace(fichier)


# ───────────────────────── Mesure du timbre ─────────────────────────

# Plage de recherche de la hauteur selon la voix : limiter la plage évite les « sauts d'octave »
# (une voix grave mesurée au double de sa fréquence réelle).
BORNES_F0 = {"enfant": (150, 600), "femme": (120, 450), "homme": (60, 260), None: (60, 500)}


def bornes_f0(sexe: str | None, age: str | None = None) -> tuple[int, int]:
    return BORNES_F0["enfant"] if age == "enfant" else BORNES_F0.get(sexe, BORNES_F0[None])


def mesurer_f0(wav: Path, debut: float = 0.0, duree: float | None = None,
               bornes: tuple[int, int] = BORNES_F0[None]) -> float | None:
    """Fréquence fondamentale médiane (Hz) d'un enregistrement, sur les parties voisées."""
    import librosa

    signal, taux = librosa.load(str(wav), sr=16000, mono=True, offset=debut, duration=duree)
    if signal.size < 16000 * 0.5:
        return None
    f0, voise, _ = librosa.pyin(signal, fmin=bornes[0], fmax=bornes[1], sr=taux, frame_length=1024)
    valeurs = f0[voise & ~np.isnan(f0)]
    return round(float(np.median(valeurs)), 1) if valeurs.size >= 10 else None


def registre(sexe: str, f0: float | None) -> str | None:
    if f0 is None:
        return None
    graves, aigus = (180, 230) if sexe == "femme" else (105, 140)
    return "grave" if f0 < graves else "aigu" if f0 > aigus else "médium"


def sexe_estime(f0: float | None) -> Sexe | None:
    return None if f0 is None else ("homme" if f0 < SEUIL_SEXE_HZ else "femme")


# ───────────────────────── Voix personnelles ─────────────────────────

def ajouter_voix_perso(nom: str, sexe: str, age: str, echantillon: Path) -> dict[str, Any]:
    """Ajoute un échantillon de voix (converti en WAV mono 24 kHz) à la bibliothèque."""
    import subprocess

    from pipeline.extract import FFmpegError, _require

    voix_id = f"perso-{uuid.uuid4().hex[:8]}"
    fichier = f"perso/{voix_id}.wav"
    cible = dossier_voix() / fichier
    cible.parent.mkdir(parents=True, exist_ok=True)
    resultat = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-i", str(echantillon), "-t", "20", "-ac", "1", "-ar", "24000",
         "-af", "highpass=f=70", str(cible)],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise FFmpegError("Fichier audio illisible : utilisez un enregistrement WAV, MP3 ou M4A.")
    f0 = mesurer_f0(cible, bornes=bornes_f0(sexe, age))
    voix = {
        "id": voix_id, "nom": nom.strip() or "Ma voix", "source": "perso", "fichier": fichier,
        "sexe": sexe, "age": age, "registre": registre(sexe, f0), "f0_hz": f0,
    }
    # L'échantillon sert aussi d'aperçu
    fichier_apercu(voix_id).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cible, fichier_apercu(voix_id))
    banque = lire_banque()
    banque.append(voix)
    ecrire_banque(banque)
    return {**voix, "apercu": True}


def supprimer_voix_perso(voix_id: str) -> None:
    banque = lire_banque()
    voix = next((v for v in banque if v["id"] == voix_id), None)
    if voix is None or voix["source"] != "perso":
        raise KeyError(voix_id)
    (dossier_voix() / voix["fichier"]).unlink(missing_ok=True)
    fichier_apercu(voix_id).unlink(missing_ok=True)
    ecrire_banque([v for v in banque if v["id"] != voix_id])


# ───────────────────────── Aperçus des voix XTTS ─────────────────────────

def generer_apercus() -> int:
    """Fait lire une phrase d'exemple à chaque voix XTTS sans aperçu, puis mesure son registre."""
    from pipeline.voice import fichier_brut, generer_voix

    banque = lire_banque()
    manquantes = [v for v in banque if v["source"] == "xtts" and not fichier_apercu(v["id"]).is_file()]
    if not manquantes:
        return 0
    dossier = dossier_voix() / "apercus"
    dossier.mkdir(parents=True, exist_ok=True)
    phrases = [
        {"id": n, "debut": 0.0, "fin": 12.0, "fin_max": 12.0, "traduction": TEXTE_APERCU}
        for n in range(len(manquantes))
    ]
    generer_voix(phrases, "fr", {n: spec(v) for n, v in enumerate(manquantes)}, dossier)
    for n, voix in enumerate(manquantes):
        fichier_brut(dossier, n).replace(fichier_apercu(voix["id"]))
        voix["f0_hz"] = mesurer_f0(fichier_apercu(voix["id"]), bornes=bornes_f0(voix["sexe"], voix["age"]))
        voix["registre"] = registre(voix["sexe"], voix["f0_hz"])
    (dossier / "manifeste.json").unlink(missing_ok=True)
    ecrire_banque(banque)
    return len(manquantes)


# ───────────────────────── Attribution automatique ─────────────────────────

def attribuer(locuteurs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Choisit pour chaque locuteur une voix différente de la bibliothèque, du même sexe et du
    registre le plus proche de sa voix d'origine (f0_hz). Renvoie {code locuteur: id de voix}."""
    banque = lire_banque()
    prises: set[str] = set()
    choix: dict[str, str] = {}
    # Les locuteurs dont la voix est mesurée choisissent en premier
    for code, infos in sorted(locuteurs.items(), key=lambda kv: kv[1].get("f0_hz") is None):
        f0 = infos.get("f0_hz")
        sexe = sexe_estime(f0)
        # Voix d'enfant seulement pour une voix d'origine très aiguë ; voix inconnue : adulte
        ages = {"adulte"} if f0 is None else ({"enfant"} if f0 >= SEUIL_ENFANT_HZ else {"jeune", "adulte", "senior"})
        libres = [v for v in banque if v["id"] not in prises]
        candidates = (
            [v for v in libres if v["age"] in ages and (sexe is None or v["sexe"] == sexe)]
            or [v for v in libres if v["age"] != "enfant" and (sexe is None or v["sexe"] == sexe)]
            or libres
            or banque
        )
        if f0 is not None:
            # L'âge ne se mesure pas : à hauteur proche, une voix d'adulte est préférée
            candidates.sort(key=lambda v: abs((v.get("f0_hz") or (200 if v["sexe"] == "femme" else 120)) - f0)
                            + (0 if v["age"] in ("adulte", "enfant") else PENALITE_AGE_HZ))
        choix[code] = candidates[0]["id"]
        prises.add(candidates[0]["id"])
    return choix


def remesurer() -> None:
    """Remesure la hauteur et le registre de toutes les voix à partir de leurs aperçus existants."""
    banque = lire_banque()
    for voix in banque:
        if fichier_apercu(voix["id"]).is_file():
            voix["f0_hz"] = mesurer_f0(fichier_apercu(voix["id"]), bornes=bornes_f0(voix["sexe"], voix["age"]))
            voix["registre"] = registre(voix["sexe"], voix["f0_hz"])
    ecrire_banque(banque)
