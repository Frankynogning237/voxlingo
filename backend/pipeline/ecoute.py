"""Écoute de contrôle des voix doublées : chaque phrase générée dit-elle bien son texte ?

Le moteur de clonage (XTTS, Chatterbox) se trompe parfois : mot sauté ou déformé, phrase bafouillée,
syllabes inventées ou bruit ajouté en fin de phrase. Après la génération, chaque phrase est retranscrite
par Whisper dans la langue cible et comparée, mot à mot, au texte qu'elle devait dire. Une phrase nettement
infidèle est refaite une fois (pipeline/runner.py), et la version la plus fidèle des deux est gardée.

Le modèle est chargé pour la vérification puis libéré ; les phrases sont lues une à une (mémoire bornée).
"""

from __future__ import annotations

import gc
import logging
import re
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger("voxlingo.ecoute")

FIDELITE_MIN = 0.65  # en dessous, la phrase ne dit pas son texte (mots sautés, déformés ou inventés)
MOTS_MIN = 3  # phrase trop courte : la retranscription d'un ou deux mots n'est pas assez fiable
PART_REPRISE_MAX = 0.4  # au plus 40 % des phrases refaites : au-delà, c'est la vérification qui doute

Ecoute = Callable[[Path, str], str]


def _mots(texte: str) -> list[str]:
    """Mots comparables : minuscules, sans accents ni ponctuation ; les nombres sont ignorés (Whisper écrit
    « 3 » là où le texte dit « drei », et inversement)."""
    texte = unicodedata.normalize("NFKD", texte.casefold())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return [m for m in re.findall(r"\w+", texte) if not any(c.isdigit() for c in m)]


def fidelite(attendu: str, entendu: str) -> float:
    """Ressemblance (0 à 1) entre le texte à dire et ce qui a été entendu : mots manquants, en trop ou
    déformés la font baisser."""
    a, b = _mots(attendu), _mots(entendu)
    if not a:
        return 1.0
    return round(SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)


def a_verifier(texte: str) -> bool:
    return len(_mots(texte)) >= MOTS_MIN


def phrases_a_refaire(fidelites: dict[int, float]) -> set[int]:
    """Phrases nettement infidèles, les moins fidèles d'abord, au plus PART_REPRISE_MAX des phrases (au moins 2)."""
    suspectes = sorted((f, i) for i, f in fidelites.items() if f < FIDELITE_MIN)
    return {i for _, i in suspectes[: max(2, int(len(fidelites) * PART_REPRISE_MAX))]}


@contextmanager
def modele_ecoute() -> Iterator[Ecoute | None]:
    """Fonction (fichier WAV, langue) → texte entendu ; None si Whisper est indisponible. Sur processeur,
    le modèle « small » suffit pour des phrases courtes et nettes ; sur GPU, celui de la transcription."""
    try:
        from config import get_runtime
        from pipeline.transcribe import _charger_16k, charger_modele

        runtime = get_runtime()
        nom = runtime.whisper_model if runtime.device == "cuda" else "small"
        modele = charger_modele(nom, runtime.device, runtime.whisper_compute_type)
    except Exception as exc:  # noqa: BLE001 — facultatif : le doublage marche sans
        logger.warning("Écoute de contrôle indisponible (%s)", exc)
        yield None
        return

    def ecouter(fichier: Path, langue: str) -> str:
        segments, _ = modele.transcribe(_charger_16k(fichier), language=langue, beam_size=3, vad_filter=False,
                                        condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segments)

    try:
        yield ecouter
    finally:
        del modele
        gc.collect()


def mesurer(fichiers: dict[int, Path], textes: dict[int, str], langue: str, ecouter: Ecoute,
            on_progress: Callable[[float], None] | None = None) -> dict[int, float]:
    """Fidélité de chaque phrase générée à son texte (phrases assez longues seulement)."""
    a_lire = [i for i in fichiers if fichiers[i].is_file() and a_verifier(textes.get(i, ""))]
    resultat: dict[int, float] = {}
    for n, ident in enumerate(a_lire):
        try:
            resultat[ident] = fidelite(textes[ident], ecouter(fichiers[ident], langue))
        except Exception as exc:  # noqa: BLE001 — une phrase illisible ne bloque pas les autres
            logger.warning("Écoute de la phrase %d impossible (%s)", ident, exc)
        if on_progress:
            on_progress((n + 1) / len(a_lire))
    return resultat
