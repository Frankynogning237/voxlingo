"""Ressemblance vocale : la voix clonée doit rester celle du locuteur d'origine.

Le même modèle d'empreinte vocale que l'identification des locuteurs (SpeechBrain ECAPA, voir
pipeline/diarize.py) sert à deux moments :

1. **Avant la synthèse**, pour composer l'échantillon de clonage : chaque phrase candidate du locuteur
   reçoit une empreinte, comparée à l'empreinte « type » de ce locuteur (moyenne robuste de ses
   phrases). Les extraits qui ne lui ressemblent pas (phrase attribuée à tort, voix superposée,
   musique résiduelle, cri saturé) sont écartés, et les plus représentatifs passent en tête : Chatterbox
   n'écoute que les premières secondes de l'échantillon.
2. **Après la synthèse**, pour vérifier chaque phrase générée : sa ressemblance avec l'échantillon est
   mesurée ; une phrase nettement moins ressemblante que les autres du même locuteur (timbre qui a
   dérivé) est refaite une fois en mode « stable », et la meilleure des deux versions est gardée.

Lecture par extraits (mémoire bornée, quelle que soit la durée de la vidéo). Sans le modèle (SpeechBrain
absent), tout continue comme avant : aucun extrait n'est écarté, aucune phrase n'est refaite.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("voxlingo.ressemblance")

TAUX = 16000  # fréquence attendue par ECAPA
DUREE_MIN_S = 1.0  # en dessous, l'empreinte n'est pas fiable
CANDIDATS_MAX = 40  # phrases évaluées par locuteur pour l'échantillon (les plus longues)
ECART_ECHANTILLON = 0.15  # extrait écarté s'il ressemble nettement moins que les autres au locuteur
ECART_PHRASE = 0.12  # phrase générée refaite si elle ressemble nettement moins que les autres…
SEUIL_PHRASE = 0.6  # … et pas franchement ressemblante dans l'absolu
PART_REPRISE_MAX = 0.3  # au plus 30 % des phrases d'un locuteur refaites (le reste est fidèle)

Empreinte = Callable[[np.ndarray], "np.ndarray | None"]


def _lire(fichier: Path, debut: float | None = None, fin: float | None = None) -> np.ndarray:
    import librosa

    if debut is None:
        signal, _ = librosa.load(str(fichier), sr=TAUX, mono=True)
    else:
        signal, _ = librosa.load(str(fichier), sr=TAUX, mono=True, offset=max(0.0, debut),
                                 duration=max(0.0, fin - debut))
    return signal


@contextmanager
def modele_empreinte() -> Iterator[Empreinte | None]:
    """Fonction signal 16 kHz → empreinte normée (ou None si l'extrait est trop court).
    Donne None si le modèle est indisponible ; le modèle est libéré à la sortie."""
    try:
        import torch

        from pipeline.diarize import _modele

        modele = _modele()
    except Exception as exc:  # noqa: BLE001 — facultatif : le clonage marche sans
        logger.warning("Empreintes vocales indisponibles (%s) : ressemblance non vérifiée", exc)
        yield None
        return

    def empreinte(signal: np.ndarray) -> np.ndarray | None:
        if signal.size < DUREE_MIN_S * TAUX:
            return None
        with torch.no_grad():
            vecteur = modele.encode_batch(torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
                                          .unsqueeze(0)).squeeze().cpu().numpy()
        return vecteur / (np.linalg.norm(vecteur) or 1.0)

    try:
        yield empreinte
    finally:
        del modele
        gc.collect()


def centre_robuste(empreintes: list[np.ndarray]) -> np.ndarray:
    """Empreinte « type » : moyenne, puis moyenne des 70 % les plus proches (les intrus pèsent peu)."""
    tableau = np.stack(empreintes)
    centre = tableau.mean(axis=0)
    centre /= np.linalg.norm(centre) or 1.0
    if len(tableau) >= 3:
        proches = np.argsort(tableau @ centre)[::-1][: max(2, int(round(len(tableau) * 0.7)))]
        centre = tableau[proches].mean(axis=0)
        centre /= np.linalg.norm(centre) or 1.0
    return centre


def scores_candidats(source: Path, segments: list[dict[str, Any]], empreinte: Empreinte) -> dict[int, float]:
    """Ressemblance (cosinus) de chaque phrase candidate avec l'empreinte type du locuteur.

    Les phrases trop différentes des autres reçoivent un score sous `seuil_echantillon(scores)`."""
    candidats = sorted((s for s in segments if s["fin"] - s["debut"] >= DUREE_MIN_S),
                       key=lambda s: s["fin"] - s["debut"], reverse=True)[:CANDIDATS_MAX]
    vecteurs: dict[int, np.ndarray] = {}
    for seg in candidats:
        vecteur = empreinte(_lire(source, seg["debut"], seg["fin"]))
        if vecteur is not None:
            vecteurs[seg["id"]] = vecteur
    if len(vecteurs) < 2:
        return {}  # rien à comparer
    centre = centre_robuste(list(vecteurs.values()))
    return {ident: round(float(v @ centre), 3) for ident, v in vecteurs.items()}


def seuil_echantillon(scores: dict[int, float]) -> float:
    """Score sous lequel un extrait est écarté de l'échantillon : nettement sous la médiane du locuteur."""
    return float(np.median(list(scores.values()))) - ECART_ECHANTILLON if scores else -1.0


def phrases_a_refaire(ressemblances: dict[int, float]) -> set[int]:
    """Phrases générées dont le timbre a dérivé : nettement sous la médiane du locuteur, et pas franchement
    ressemblantes dans l'absolu. Au plus PART_REPRISE_MAX des phrases, les moins ressemblantes d'abord."""
    if len(ressemblances) < 3:
        return set()
    mediane = float(np.median(list(ressemblances.values())))
    suspectes = sorted((s, i) for i, s in ressemblances.items() if s < mediane - ECART_PHRASE and s < SEUIL_PHRASE)
    return {i for _, i in suspectes[: int(len(ressemblances) * PART_REPRISE_MAX)]}


def ressemblance_fichier(fichier: Path, reference: np.ndarray, empreinte: Empreinte) -> float | None:
    if not fichier.is_file():
        return None
    vecteur = empreinte(_lire(fichier))
    return None if vecteur is None else round(float(vecteur @ reference), 3)
