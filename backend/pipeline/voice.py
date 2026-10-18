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
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import get_runtime, get_settings
from pipeline.extract import _require
from pipeline.segments import vitesse_xtts
from pipeline.prosody import TEMPERATURE
from pipeline.voice_bank import separer_spec, transformer

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

MARGE_CHEVAUCHEMENT_S = 0.2  # un extrait qui touche la parole d'un autre locuteur n'est pas « propre »
NIVEAU_REFERENCE_DB = -20.0  # niveau commun des extraits de l'échantillon (voix ni trop faible, ni saturée)


def choisir_extraits(segments: list[dict[str, Any]], autres: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Phrases du locuteur qui composeront l'échantillon de clonage (6 à 15 s, dans l'ordre du film).

    On privilégie les phrases de 1,5 à 12 s où **personne d'autre ne parle** (`autres` : phrases des
    autres locuteurs) : une voix superposée brouillerait l'empreinte et donc le timbre cloné. À défaut,
    les phrases qui chevauchent sont acceptées, puis toutes les phrases."""
    def propre(seg: dict[str, Any]) -> bool:
        return not any(a["debut"] < seg["fin"] + MARGE_CHEVAUCHEMENT_S and a["fin"] > seg["debut"] - MARGE_CHEVAUCHEMENT_S
                       for a in autres or [])

    longueur = lambda s: s["fin"] - s["debut"]  # noqa: E731
    bonnes = [s for s in segments if 1.5 <= longueur(s) <= 12]
    candidats = (sorted((s for s in bonnes if propre(s)), key=longueur, reverse=True)
                 + sorted((s for s in bonnes if not propre(s)), key=longueur, reverse=True)
                 ) or sorted(segments, key=longueur, reverse=True)
    choisis, total = [], 0.0
    for seg in candidats:
        if total >= REFERENCE_MIN_S + 2 or total + longueur(seg) > REFERENCE_MAX_S:
            continue
        choisis.append(seg)
        total += longueur(seg)
    return sorted(choisis, key=lambda s: s["debut"])


def extraire_reference(video: Path, segments: list[dict[str, Any]], sortie: Path,
                       autres: list[dict[str, Any]] | None = None) -> Path:
    """Assemble 6 à 15 s de voix d'origine du locuteur en un échantillon propre, au niveau homogène."""
    import soundfile as sf

    from pipeline.niveau import niveau_actif_db

    choisis = choisir_extraits(segments, autres)
    if not choisis:
        raise VoiceError("Aucune parole exploitable pour cloner la voix.")

    morceaux = []
    for seg in choisis:
        resultat = subprocess.run(
            [_require("ffmpeg"), "-v", "error", "-ss", f"{seg['debut']:.2f}", "-to", f"{seg['fin']:.2f}",
             "-i", str(video), "-vn", "-ac", "1", "-ar", str(TAUX), "-af", "highpass=f=70",
             "-f", "s16le", "-"],
            capture_output=True,
        )
        if resultat.returncode == 0 and resultat.stdout:
            extrait = np.frombuffer(resultat.stdout, dtype=np.int16).astype(np.float32) / 32768
            niveau = niveau_actif_db(extrait, TAUX)
            if niveau is not None:  # chaque extrait au même niveau : une empreinte vocale cohérente
                extrait = extrait * 10 ** ((NIVEAU_REFERENCE_DB - niveau) / 20)
            morceaux.append(np.clip(extrait, -0.98, 0.98))
            morceaux.append(np.zeros(int(0.25 * TAUX), dtype=np.float32))  # courte pause entre extraits
    if not morceaux:
        raise VoiceError("Impossible d'extraire l'échantillon de voix d'origine.")
    sf.write(sortie, np.concatenate(morceaux), TAUX, subtype="PCM_16")
    logger.info("Échantillon de référence : %.1f s (%d extraits)",
                sum(s["fin"] - s["debut"] for s in choisis), len(choisis))
    return sortie


def temperature(seg: dict[str, Any]) -> float:
    """Expressivité de XTTS : selon le ton de la phrase, renforcée par une émotion forte."""
    base = TEMPERATURE.get(seg.get("ton") or "neutre", 0.7)
    emotion = seg.get("emotion") or {}
    if emotion.get("emotion") in ("colère", "joie", "surprise", "peur") and emotion.get("intensite") == "forte":
        base += 0.1
    elif emotion.get("emotion") == "tristesse":
        base -= 0.05
    return round(min(0.9, max(0.5, base)), 2)


INTONATION_MIN_S, INTONATION_MAX_S = 1.2, 15.0  # extrait de phrase exploitable pour l'intonation


def extraire_intonation(source: Path, seg: dict[str, Any], sortie: Path) -> Path | None:
    """Extrait la phrase d'origine (voix isolée) qui guidera l'intonation de sa traduction."""
    duree = seg["fin"] - seg["debut"]
    if not INTONATION_MIN_S <= duree <= INTONATION_MAX_S:
        return None
    if not sortie.is_file():
        resultat = subprocess.run(
            [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{seg['debut']:.2f}", "-to", f"{seg['fin']:.2f}",
             "-i", str(source), "-vn", "-ac", "1", "-ar", str(TAUX), "-af", "highpass=f=70", str(sortie)],
            capture_output=True,
        )
        if resultat.returncode != 0 or not sortie.is_file():
            return None
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

    CHAMPS = ("texte", "langue", "disponible", "voix", "ton", "intonation")

    @staticmethod
    def _cle(seg: dict[str, Any], langue: str, voix: str) -> dict[str, Any]:
        # Le temps disponible fixe la vitesse de lecture ; la voix choisie, le timbre :
        # si l'un des deux change, la phrase est refaite
        disponible = round(seg.get("fin_max", seg["fin"]) - seg["debut"], 2)
        return {"texte": seg["traduction"], "langue": langue, "disponible": disponible, "voix": voix,
                "ton": seg.get("ton"), "intonation": True if seg.get("intonation") else None}

    def a_jour(self, seg: dict[str, Any], langue: str, voix: str, fichier: Path) -> bool:
        entree = self.donnees.get(str(seg["id"]), {})
        return fichier.is_file() and {k: entree.get(k) for k in self.CHAMPS} == self._cle(seg, langue, voix)

    def noter(self, seg: dict[str, Any], langue: str, voix: str, vitesse: float | None = None) -> None:
        self.donnees[str(seg["id"])] = {**self._cle(seg, langue, voix), "vitesse": vitesse}
        self.chemin.write_text(json.dumps(self.donnees, ensure_ascii=False, indent=1), encoding="utf-8")


def fichier_brut(dossier: Path, seg_id: int) -> Path:
    return dossier / f"brut_{seg_id:05d}.wav"


# ───────────────────────── Chatterbox (voix plus expressive) ─────────────────────────

# Langues de Chatterbox Multilingual (codes du moteur)
LANGUES_CHATTERBOX = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
                      "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh-cn"}
_chatterbox_pret = False


def chatterbox_disponible() -> bool:
    """Vrai si l'environnement séparé de Chatterbox est installé (réglage CHATTERBOX_PYTHON).

    Sur Kaggle il s'installe en arrière-plan : on revérifie à chaque génération tant qu'il n'est pas prêt."""
    global _chatterbox_pret
    python = get_settings().chatterbox_python
    if _chatterbox_pret or not python or not Path(python).is_file():
        return _chatterbox_pret
    try:
        _chatterbox_pret = subprocess.run([python, "-c", "import chatterbox.mtl_tts"],
                                          capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
    return _chatterbox_pret


def moteur_voix(langue: str) -> str:
    """« chatterbox » ou « xtts », selon le réglage, la langue et ce qui est installé."""
    from config import lire_parametres

    choix = lire_parametres().get("moteur_voix", "auto")
    if choix == "xtts" or langue not in LANGUES_CHATTERBOX:
        return "xtts"
    if chatterbox_disponible():
        return "chatterbox"
    if choix == "chatterbox":
        logger.warning("Chatterbox demandé mais pas encore installé : XTTS v2 est utilisé.")
    return "xtts"


def voix_pour_chatterbox(voix: dict[int, str]) -> dict[int, str] | None:
    """Chatterbox ne clone qu'un échantillon : une voix intégrée à XTTS (« xtts:<nom> ») est remplacée
    par son aperçu enregistré. None si une voix n'a pas encore d'aperçu (XTTS sera utilisé)."""
    from pipeline import voice_bank

    apercus = {v["xtts"]: voice_bank.fichier_apercu(v["id"]) for v in voice_bank.lire_banque() if v.get("xtts")}
    converties = {}
    for ident, spec in voix.items():
        base, sep, transformation = spec.partition("|")
        if base.startswith("xtts:"):
            apercu = apercus.get(base[5:])
            if apercu is None or not apercu.is_file():
                logger.info("Pas d'aperçu pour la voix %s : XTTS v2 est utilisé pour ce projet.", base[5:])
                return None
            base = f"ref:{apercu}"
        converties[ident] = base + sep + transformation
    return converties


MEMOIRE_CONSEILLEE_GO = 3.0  # XTTS v2 sur processeur

# Une seule génération XTTS à la fois : deux processus de 3 à 4 Go saturent la mémoire.
# Les suivantes attendent leur tour (projets, aperçus de la bibliothèque).
_file_xtts = threading.Lock()


def _memoire_libre_go() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return float("nan")


def generer_voix(
    segments: list[dict[str, Any]],
    langue: str,
    voix: dict[int, str],
    dossier: Path,
    on_progress: Callable[[float], None] | None = None,
    moteur: str | None = None,
) -> int:
    """Synthétise chaque segment traduit avec sa voix. Renvoie le nombre de phrases générées.

    `voix` associe à chaque identifiant de segment une voix : « ref:<chemin WAV> » (voix
    d'origine clonée, ou voix personnelle de la bibliothèque) ou « xtts:<nom> » (voix
    intégrée à XTTS v2, voir pipeline/voice_bank.py).

    La synthèse tourne dans un processus séparé (pipeline/voice_worker.py pour XTTS v2,
    pipeline/chatterbox_worker.py pour Chatterbox) : un manque de mémoire n'arrête que ce
    processus, et devient un message d'erreur lisible. Si Chatterbox échoue, XTTS prend le relais.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    voix_origine = voix
    moteur = moteur or moteur_voix(langue)
    if moteur == "chatterbox":
        converties = voix_pour_chatterbox({s["id"]: voix[s["id"]] for s in segments})
        if converties is None:
            moteur = "xtts"
        else:
            voix = converties
    # Le moteur fait partie de la voix : en changer régénère les phrases
    cles = {s["id"]: voix[s["id"]] if moteur == "xtts" else f"chatterbox#{voix[s['id']]}" for s in segments}
    manifeste = Manifeste(dossier)
    a_generer = [s for s in segments if not manifeste.a_jour(s, langue, cles[s["id"]], fichier_brut(dossier, s["id"]))]
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
                "voix": voix[s["id"]],
                "temperature": temperature(s),
                "intonation": s.get("intonation"),
            }
            for s in a_generer
        ],
        "langue": langue,
        "dossier": str(dossier),
    }, ensure_ascii=False), encoding="utf-8")
    par_id = {s["id"]: s for s in a_generer}

    if not _file_xtts.acquire(blocking=False):
        logger.info("Une autre génération de voix est en cours : en attente de son tour")
        while not _file_xtts.acquire(timeout=2):
            if on_progress:
                on_progress(0.0)  # point de contrôle : permet d'annuler pendant l'attente
    try:
        if moteur == "chatterbox":
            commande = [get_settings().chatterbox_python, "-X", "utf8",
                        str(Path(__file__).resolve().parent / "chatterbox_worker.py"), str(tache)]
        else:
            commande = [sys.executable, "-X", "utf8", "-m", "pipeline.voice_worker", str(tache)]
        try:
            return _lancer_xtts(commande, moteur, tache, a_generer, par_id, manifeste, langue, voix, cles, dossier,
                                libre, on_progress)
        except VoiceError as exc:
            if moteur != "chatterbox":
                raise
            logger.warning("Chatterbox a échoué (%s) : XTTS v2 prend le relais.", exc)
    finally:
        _file_xtts.release()
    return generer_voix(segments, langue, voix_origine, dossier, on_progress, moteur="xtts")


def _lancer_xtts(commande: list[str], moteur: str, tache: Path, a_generer: list[dict[str, Any]],
                 par_id: dict[int, dict[str, Any]], manifeste: "Manifeste", langue: str, voix: dict[int, str],
                 cles: dict[int, str], dossier: Path, libre: float,
                 on_progress: Callable[[float], None] | None) -> int:
    """Lance le processus de synthèse (XTTS ou Chatterbox) et suit sa progression."""
    nom = "Chatterbox" if moteur == "chatterbox" else "XTTS v2"
    logger.info("Chargement de %s dans un processus séparé (%s)", nom, get_runtime().device)
    processus = subprocess.Popen(
        commande,
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
                logger.info("%s chargé, génération de %d phrase(s)", nom, len(a_generer))
            elif ligne.startswith("FAIT "):
                _, ident, vitesse = ligne.split()
                seg = par_id[int(ident)]
                _, transformation = separer_spec(voix[seg["id"]])
                transformer(fichier_brut(dossier, seg["id"]), transformation)  # voix d'enfant, de senior…
                manifeste.noter(seg, langue, cles[seg["id"]], float(vitesse))
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
        logger.error("Processus %s terminé avec le code %s : %s", nom, processus.returncode, erreurs[-1500:])
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
