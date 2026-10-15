"""Identification des locuteurs : qui parle dans chaque phrase.

100 % local et sans compte : chaque phrase reçoit une « empreinte vocale » (modèle
public SpeechBrain ECAPA-TDNN, ~80 Mo), puis les empreintes proches sont regroupées
(classification hiérarchique, distance cosinus). Le nombre de locuteurs est trouvé
automatiquement. Les phrases trop courtes pour une empreinte fiable prennent le
locuteur de la phrase voisine la plus proche dans le temps.

Le modèle pyannote prévu à l'origine exige un compte Hugging Face et l'acceptation
de ses conditions ; cette approche évite cette démarche.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.diarize")

TAUX = 16000
DUREE_MIN_EMPREINTE_S = 0.8  # en dessous, l'empreinte vocale n'est pas fiable
SEUIL_DISTANCE = 0.62  # distance cosinus moyenne au-delà de laquelle deux groupes sont des personnes différentes
COULEURS = ["#a855f7", "#3b82f6", "#22c55e", "#f97316", "#ec4899", "#eab308", "#14b8a6", "#ef4444"]


def _lire_extrait(chemin: Path, debut: float, fin: float) -> np.ndarray:
    """Extrait [debut, fin] d'un WAV 16 kHz mono, en mémoire bornée."""
    with wave.open(str(chemin), "rb") as wav:
        taux = wav.getframerate()
        wav.setpos(min(int(debut * taux), wav.getnframes()))
        brut = wav.readframes(max(0, int((fin - debut) * taux)))
        canaux = wav.getnchannels()
    signal = np.frombuffer(brut, dtype=np.int16).astype(np.float32) / 32768
    if canaux > 1:
        signal = signal.reshape(-1, canaux).mean(axis=1)
    return signal


def _modele():
    if sys.platform == "win32":
        dll = Path(sys.prefix) / "Library" / "bin"
        if dll.is_dir():
            os.add_dll_directory(str(dll))
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(get_settings().models_dir / "speechbrain" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": get_runtime().device},
        # Windows refuse les liens symboliques sans droits administrateur : on copie les fichiers
        local_strategy=LocalStrategy.COPY,
    )


def identifier_locuteurs(
    segments: list[dict[str, Any]],
    audio_16k: Path,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, dict[str, str]]:
    """Ajoute `locuteur` (« SPEAKER_00 »…) à chaque segment et renvoie la fiche des locuteurs."""
    import torch
    from sklearn.cluster import AgglomerativeClustering

    if not segments:
        return {}
    modele = _modele()
    empreintes: dict[int, np.ndarray] = {}
    try:
        for n, seg in enumerate(segments):
            if seg["fin"] - seg["debut"] >= DUREE_MIN_EMPREINTE_S:
                extrait = _lire_extrait(audio_16k, seg["debut"], seg["fin"])
                with torch.no_grad():
                    vecteur = modele.encode_batch(torch.from_numpy(extrait).unsqueeze(0)).squeeze().cpu().numpy()
                empreintes[n] = vecteur / (np.linalg.norm(vecteur) or 1.0)
            if on_progress:
                on_progress(0.9 * (n + 1) / len(segments))
    finally:
        del modele
        gc.collect()

    indices = sorted(empreintes)
    if len(indices) >= 2:
        groupes = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=SEUIL_DISTANCE
        ).fit_predict(np.stack([empreintes[i] for i in indices]))
    else:
        groupes = np.zeros(len(indices), dtype=int)
    groupe_de = dict(zip(indices, groupes.tolist()))

    # Phrases trop courtes : locuteur de la phrase mesurée la plus proche dans le temps
    for n, seg in enumerate(segments):
        if n not in groupe_de:
            if indices:
                proche = min(indices, key=lambda i: abs(segments[i]["debut"] - seg["debut"]))
                groupe_de[n] = groupe_de[proche]
            else:
                groupe_de[n] = 0

    # Numérotation dans l'ordre d'apparition : SPEAKER_00 parle en premier
    ordre: dict[int, str] = {}
    for n in range(len(segments)):
        ordre.setdefault(groupe_de[n], f"SPEAKER_{len(ordre):02d}")
        segments[n]["locuteur"] = ordre[groupe_de[n]]

    locuteurs = {
        code: {"nom": f"Locuteur {k + 1}", "couleur": COULEURS[k % len(COULEURS)]}
        for k, code in enumerate(ordre.values())
    }
    logger.info("%d locuteur(s) identifié(s) sur %d phrases", len(locuteurs), len(segments))
    if on_progress:
        on_progress(1.0)
    return locuteurs
