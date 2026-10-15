"""Transcription et détection de langue avec faster-whisper.

Le modèle est chargé pour la tâche puis déchargé, pour libérer la mémoire
avant les étapes suivantes. L'horodatage au mot via WhisperX arrive en phase 6.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.transcribe")


@dataclass
class Segment:
    id: int
    debut: float
    fin: float
    texte: str


@dataclass
class Transcription:
    langue: str
    probabilite: float
    duree_s: float
    segments: list[Segment]

    def to_dict(self) -> dict:
        return asdict(self)


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
        segments_iter, info = model.transcribe(str(audio), vad_filter=True, beam_size=5)
        logger.info("Langue détectée : %s (%.0f %%)", info.language, info.language_probability * 100)

        segments: list[Segment] = []
        for i, seg in enumerate(segments_iter):
            segments.append(Segment(id=i, debut=round(seg.start, 2), fin=round(seg.end, 2),
                                    texte=seg.text.strip()))
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))

        return Transcription(
            langue=info.language,
            probabilite=round(info.language_probability, 3),
            duree_s=round(info.duration, 2),
            segments=segments,
        )
    finally:
        del model
        gc.collect()
        if runtime.device == "cuda":
            import torch

            torch.cuda.empty_cache()
