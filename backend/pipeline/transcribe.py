"""Transcription et détection de langue avec faster-whisper.

Le modèle est chargé pour la tâche puis déchargé, pour libérer la mémoire
avant les étapes suivantes. Le minutage de chaque mot est conservé : il sert
à redécouper précisément les phrases trop longues (pipeline/segments.py).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.transcribe")


@dataclass
class Segment:
    id: int
    debut: float
    fin: float
    texte: str
    mots: list[dict] = field(default_factory=list)  # [{"mot", "debut", "fin"}]


@dataclass
class Transcription:
    langue: str
    probabilite: float
    duree_s: float
    segments: list[Segment]

    def to_dict(self) -> dict:
        return asdict(self)


FIN_DE_PHRASE = (".", "!", "?", "…", ":", ";", "。", "！", "？")
ECART_MAX_S = 0.6  # silence maximal entre deux morceaux d'une même phrase
DUREE_MAX_S = 15.0  # au-delà, on garde les segments séparés


def fusionner_fragments(segments: list[Segment]) -> list[Segment]:
    """Recolle les phrases coupées par Whisper (« …into another » + « language. »).

    Traduire une phrase entière donne une traduction bien meilleure qu'un morceau isolé.
    """
    fusion: list[Segment] = []
    for seg in segments:
        prec = fusion[-1] if fusion else None
        if (
            prec is not None
            and not prec.texte.rstrip().endswith(FIN_DE_PHRASE)
            and seg.debut - prec.fin <= ECART_MAX_S
            and seg.fin - prec.debut <= DUREE_MAX_S
        ):
            prec.fin = seg.fin
            prec.texte = f"{prec.texte} {seg.texte}".strip()
            prec.mots.extend(seg.mots)
        else:
            fusion.append(Segment(id=len(fusion), debut=seg.debut, fin=seg.fin, texte=seg.texte,
                                  mots=list(seg.mots)))
    return fusion


def _charger_16k(audio: Path):
    """Signal mono 16 kHz (float32) attendu par Whisper."""
    import librosa
    import numpy as np

    signal, _ = librosa.load(str(audio), sr=16000, mono=True)
    return signal.astype(np.float32)


def transcribe(audio: Path, on_progress: Callable[[float], None] | None = None) -> Transcription:
    """Transcrit l'audio et détecte sa langue. `on_progress` reçoit une valeur entre 0 et 1."""
    from faster_whisper import WhisperModel

    runtime = get_runtime()
    logger.info("Chargement de Whisper %s (%s, %s)", runtime.whisper_model, runtime.device,
                runtime.whisper_compute_type)
    model = WhisperModel(
        runtime.whisper_model,
        device=runtime.device,
        compute_type=runtime.whisper_compute_type,
        download_root=str(get_settings().models_dir / "whisper"),
    )
    try:
        # Le son est lu ici (WAV via soundfile) plutôt que par faster-whisper, qui passe par PyAV :
        # certaines versions de PyAV (Kaggle) refusent son option « metadata_errors ».
        segments_iter, info = model.transcribe(
            _charger_16k(audio), vad_filter=True, beam_size=5, word_timestamps=True
        )
        logger.info("Langue détectée : %s (%.0f %%)", info.language, info.language_probability * 100)

        segments: list[Segment] = []
        for i, seg in enumerate(segments_iter):
            mots = [
                {"mot": w.word.strip(), "debut": round(w.start, 2), "fin": round(w.end, 2)}
                for w in (seg.words or [])
                if w.word.strip()
            ]
            segments.append(Segment(id=i, debut=round(seg.start, 2), fin=round(seg.end, 2),
                                    texte=seg.text.strip(), mots=mots))
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))

        return Transcription(
            langue=info.language,
            probabilite=round(info.language_probability, 3),
            duree_s=round(info.duration, 2),
            segments=fusionner_fragments(segments),
        )
    finally:
        del model
        gc.collect()
        if runtime.device == "cuda":
            import torch

            torch.cuda.empty_cache()
