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
import queue
import subprocess
import sys
import threading
import time
from collections import deque
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
REFERENCE_CIBLE_S = 12.0  # durée visée : XTTS affine son empreinte jusqu'à ~30 s, Chatterbox lit le début
REFERENCE_MAX_S = 18.0
SEUIL_SILENCE = 0.01  # amplitude sous laquelle le début et la fin sont considérés silencieux
ATTENTE_CHATTERBOX_S = 20 * 60  # installation en arrière-plan sur Kaggle (≈ 5 à 10 min) : on l'attend
SONDAGE_CHATTERBOX_S = 15.0


class VoiceError(RuntimeError):
    """Erreur lisible de la génération des voix."""


# ───────────────────────── Échantillon de référence ─────────────────────────

MARGE_CHEVAUCHEMENT_S = 0.2  # un extrait qui touche la parole d'un autre locuteur n'est pas « propre »
NIVEAU_REFERENCE_DB = -20.0  # niveau visé pour l'échantillon entier (voix ni trop faible, ni saturée)
GAIN_REFERENCE_MAX_DB = 12.0  # au-delà, on remonterait surtout le bruit de fond et la musique résiduelle
CRETE_REFERENCE = 0.95


def fichier_reference(dossier: Path, code: str) -> Path:
    """Échantillon de clonage d'un locuteur. Le numéro de version change quand la composition de
    l'échantillon s'améliore : les anciens échantillons sont refaits, et les voix clonées avec eux
    régénérées (le chemin fait partie de la voix, voir Manifeste).
    v2 : extraits gardés tels quels (plus de renormalisation un à un ni d'écrêtage) ;
    v3 : extraits choisis d'après leur ressemblance vocale avec le locuteur, le plus représentatif en tête."""
    return dossier / f"reference_v3_{code}.wav"


def choisir_extraits(segments: list[dict[str, Any]], autres: list[dict[str, Any]] | None = None,
                     scores: dict[int, float] | None = None) -> list[dict[str, Any]]:
    """Phrases du locuteur qui composeront l'échantillon de clonage (≈ 12 s), la meilleure en tête.

    On privilégie les phrases de 1,5 à 12 s où **personne d'autre ne parle** (`autres` : phrases des
    autres locuteurs) : une voix superposée brouillerait l'empreinte et donc le timbre cloné. À défaut,
    les phrases qui chevauchent sont acceptées, puis toutes les phrases.

    Avec `scores` (ressemblance de chaque phrase avec la voix type du locuteur, pipeline/ressemblance.py),
    les phrases qui ne lui ressemblent pas sont écartées et les plus ressemblantes passent en tête :
    Chatterbox n'écoute que les premières secondes de l'échantillon."""
    from pipeline.ressemblance import seuil_echantillon

    def propre(seg: dict[str, Any]) -> bool:
        return not any(a["debut"] < seg["fin"] + MARGE_CHEVAUCHEMENT_S and a["fin"] > seg["debut"] - MARGE_CHEVAUCHEMENT_S
                       for a in autres or [])

    longueur = lambda s: s["fin"] - s["debut"]  # noqa: E731
    scores = scores or {}
    bonnes = [s for s in segments if 1.5 <= longueur(s) <= 12]
    if scores:
        seuil = seuil_echantillon(scores)
        fiables = [s for s in bonnes if scores.get(s.get("id"), seuil) >= seuil]  # sans score : gardée
        if sum(longueur(s) for s in fiables) >= REFERENCE_MIN_S:
            bonnes = fiables
    preference = lambda s: (scores.get(s.get("id"), 0.0), longueur(s))  # noqa: E731
    groupes = [sorted((s for s in bonnes if propre(s)), key=preference, reverse=True),
               sorted((s for s in bonnes if not propre(s)), key=preference, reverse=True)]
    if not any(groupes):
        groupes = [sorted(segments, key=longueur, reverse=True)]
    choisis, total = [], 0.0
    for groupe in groupes:
        if choisis and total >= REFERENCE_MIN_S:
            break  # assez de phrases propres : celles où un autre parle ne servent pas
        for seg in groupe:
            if total >= REFERENCE_CIBLE_S or total + longueur(seg) > REFERENCE_MAX_S:
                continue
            choisis.append(seg)
            total += longueur(seg)
    return choisis


def extraire_reference(video: Path, segments: list[dict[str, Any]], sortie: Path,
                       autres: list[dict[str, Any]] | None = None, scores: dict[int, float] | None = None) -> Path:
    """Assemble 6 à 15 s de voix d'origine du locuteur en un échantillon propre.

    Les extraits sont gardés tels quels, les uns par rapport aux autres : le moteur de clonage
    reproduit le timbre de ce qu'il entend, et renormaliser chaque extrait (ou l'écrêter) remonte le
    bruit des passages faibles et déforme la voix. Seul un gain global, borné, et sans jamais écrêter,
    amène l'échantillon entier à un niveau confortable."""
    import soundfile as sf

    from pipeline.niveau import niveau_actif_db

    choisis = choisir_extraits(segments, autres, scores)
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
            morceaux.append(np.frombuffer(resultat.stdout, dtype=np.int16).astype(np.float32) / 32768)
            morceaux.append(np.zeros(int(0.25 * TAUX), dtype=np.float32))  # courte pause entre extraits
    if not morceaux:
        raise VoiceError("Impossible d'extraire l'échantillon de voix d'origine.")
    echantillon = np.concatenate(morceaux)
    niveau = niveau_actif_db(echantillon, TAUX)
    if niveau is not None:
        gain = float(np.clip(NIVEAU_REFERENCE_DB - niveau, -GAIN_REFERENCE_MAX_DB, GAIN_REFERENCE_MAX_DB))
        echantillon = echantillon * 10 ** (gain / 20)
    crete = float(np.max(np.abs(echantillon)))
    if crete > CRETE_REFERENCE:  # plutôt baisser le tout qu'écrêter
        echantillon = echantillon * (CRETE_REFERENCE / crete)
    sf.write(sortie, echantillon, TAUX, subtype="PCM_16")
    logger.info("Échantillon de référence : %.1f s (%d extraits)",
                sum(s["fin"] - s["debut"] for s in choisis), len(choisis))
    return sortie


TEMPERATURE_STABLE = 0.55  # phrases « stabilisées » (pipeline/ressemblance.py)


def temperature(seg: dict[str, Any]) -> float:
    """Expressivité de XTTS : selon le ton de la phrase, renforcée par une émotion forte."""
    base = TEMPERATURE.get(seg.get("ton") or "neutre", 0.7)
    emotion = seg.get("emotion") or {}
    if emotion.get("emotion") in ("colère", "joie", "surprise", "peur") and emotion.get("intensite") == "forte":
        base += 0.1
    elif emotion.get("emotion") == "tristesse":
        base -= 0.05
    if seg.get("stabiliser"):  # phrase refaite car son timbre avait dérivé : diction plus sage, voix plus sûre
        base = min(base, TEMPERATURE_STABLE)
    return round(min(0.9, max(0.5, base)), 2)


def intonation(seg: dict[str, Any]) -> str | None:
    """Extrait d'origine qui guide l'intonation, sauf pour une phrase stabilisée : son style de diction
    importé pouvait justement tirer le timbre ailleurs."""
    return None if seg.get("stabiliser") else seg.get("intonation")


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

    CHAMPS = ("texte", "langue", "disponible", "voix", "ton", "intonation", "stable")

    @staticmethod
    def _cle(seg: dict[str, Any], langue: str, voix: str) -> dict[str, Any]:
        # Le temps disponible fixe la vitesse de lecture ; la voix choisie, le timbre :
        # si l'un des deux change, la phrase est refaite
        disponible = round(seg.get("fin_max", seg["fin"]) - seg["debut"], 2)
        return {"texte": seg["traduction"], "langue": langue, "disponible": disponible, "voix": voix,
                "ton": seg.get("ton"), "intonation": True if intonation(seg) else None,
                "stable": True if seg.get("stabiliser") else None}

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


def chatterbox_en_installation() -> bool:
    """Vrai si Chatterbox est en cours d'installation (Kaggle/Colab : son dossier existe, pas encore « .pret »)."""
    python = get_settings().chatterbox_python
    if not python:
        return False
    racine = Path(python).parent.parent
    return racine.is_dir() and not (racine / ".pret").is_file()


def moteur_voix(langue: str, patienter: Callable[[], None] | None = None) -> str:
    """« chatterbox » ou « xtts », selon le réglage, la langue et ce qui est installé.

    Avec `patienter` (appelé régulièrement pendant l'attente) : si Chatterbox s'installe encore, on
    l'attend (au plus ATTENTE_CHATTERBOX_S) au lieu de passer à XTTS, nettement moins fidèle à la voix
    d'origine (ressemblance mesurée 0,45 contre 0,56 sur la même vidéo)."""
    from config import lire_parametres

    choix = lire_parametres().get("moteur_voix", "auto")
    if choix == "xtts" or langue not in LANGUES_CHATTERBOX:
        return "xtts"
    if chatterbox_disponible():
        return "chatterbox"
    if patienter is not None and chatterbox_en_installation():
        logger.info("Chatterbox s'installe encore : la génération des voix l'attend (au plus %d min)",
                    ATTENTE_CHATTERBOX_S // 60)
        fin = time.monotonic() + ATTENTE_CHATTERBOX_S
        while time.monotonic() < fin and chatterbox_en_installation():
            patienter()  # annulation possible, et le projet donne signe de vie
            time.sleep(SONDAGE_CHATTERBOX_S)
        if chatterbox_disponible():
            return "chatterbox"
        logger.warning("Chatterbox n'a pas pu être installé : XTTS v2 est utilisé.")
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
# Sans aucun signe du moteur au-delà de ces délais, la génération est arrêtée avec un message lisible
# (au lieu de rester figée) : chargement du modèle (téléchargement compris), puis chaque phrase
DELAI_CHARGEMENT_S = 20 * 60
DELAI_PHRASE_S = 10 * 60

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
                "intonation": intonation(s),
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
        errors="replace",
        env={**os.environ, "TQDM_DISABLE": "1"},  # pas de barres de progression du moteur à chaque phrase
    )
    # Les deux sorties sont lues au fil de l'eau, chacune par son fil : sinon, une fois le tampon de la
    # sortie d'erreur plein (≈ 64 Ko de messages du moteur), le processus se bloque en écrivant.
    # Les messages du moteur sont recopiés dans le journal du serveur ; les dernières lignes sont gardées
    # (mémoire bornée) pour le message d'erreur.
    erreurs_recentes: deque[str] = deque(maxlen=200)

    def lire_erreurs() -> None:
        for ligne in processus.stderr:
            erreurs_recentes.append(ligne)
            if ligne.strip():
                logger.info("[%s] %s", nom, ligne.rstrip()[:300])

    evenements: queue.Queue[str | None] = queue.Queue()

    def lire_sortie() -> None:
        for ligne in processus.stdout:
            evenements.put(ligne)
        evenements.put(None)  # fin du processus

    lecteur = threading.Thread(target=lire_erreurs, daemon=True)
    lecteur.start()
    threading.Thread(target=lire_sortie, daemon=True).start()
    faits, pret, dernier_signe = 0, False, time.monotonic()
    try:
        while True:
            try:
                ligne = evenements.get(timeout=5)
            except queue.Empty:
                if on_progress:
                    on_progress(faits / len(a_generer))  # point de contrôle : « Annuler » reste possible
                delai = DELAI_PHRASE_S if pret else DELAI_CHARGEMENT_S
                if time.monotonic() - dernier_signe > delai:
                    lecteur.join(timeout=2)
                    raise VoiceError(
                        f"{nom} ne répond plus depuis {delai // 60} min "
                        f"({'phrase ' + str(faits + 1) if pret else 'chargement du modèle'}) : génération arrêtée. "
                        "Les phrases déjà générées sont conservées ; relancez « Valider et générer les voix ». "
                        "Derniers messages du moteur : " + " | ".join(l.strip() for l in list(erreurs_recentes)[-5:])
                    )
                continue
            if ligne is None:
                break
            dernier_signe = time.monotonic()
            ligne = ligne.strip()
            if ligne == "PRET":
                pret = True
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
        lecteur.join(timeout=10)
        erreurs = "".join(erreurs_recentes).strip()
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
