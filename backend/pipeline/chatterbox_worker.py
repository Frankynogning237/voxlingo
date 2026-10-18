"""Processus séparé de synthèse Chatterbox Multilingual (Resemble AI), lancé par pipeline/voice.py.

Chatterbox demande PyTorch 2.6 et transformers 5 : il vit dans son propre environnement Python
(réglage CHATTERBOX_PYTHON) et ce script n'importe rien du moteur. Plus expressif que XTTS : son
réglage d'intensité (`exaggeration`) suit le ton et l'émotion de chaque phrase d'origine.

Même protocole que pipeline/voice_worker.py. Entrée : un fichier JSON {"segments": [{"id",
"traduction", "disponible", "voix", "temperature"}], "langue", "dossier"}, où « voix » vaut
« ref:<chemin WAV> » (échantillon de la voix à cloner).
Sortie standard, une ligne par événement :
  PRET                      modèle chargé
  FAIT <id> <vitesse>       voix du segment écrite dans <dossier>/brut_<id>.wav
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

TAUX = 24000  # fréquence attendue par le moteur (celle de XTTS v2)
SEUIL_SILENCE = 0.01
ACCELERATION_MAX = 1.25  # au-delà, le calage (atempo) ne rattrape plus : on régénère plus vite
# Codes de langue du moteur → codes Chatterbox
LANGUES = {"zh-cn": "zh"}


def langue_chatterbox(langue: str) -> str:
    return LANGUES.get(langue, langue)


def intensite(temperature: float) -> float:
    """Température XTTS (0,5 calme … 0,9 très expressif) → intensité Chatterbox (0,3 … 0,9)."""
    return round(min(0.9, max(0.3, 0.3 + (temperature - 0.5) * 1.5)), 2)


def rogner_silences(wav: np.ndarray, taux: int = TAUX) -> np.ndarray:
    """Retire les silences du début et de la fin (garde 50 ms)."""
    actif = np.flatnonzero(np.abs(wav) > SEUIL_SILENCE)
    if actif.size == 0:
        return wav
    marge = int(0.05 * taux)
    return wav[max(0, actif[0] - marge): min(len(wav), actif[-1] + marge)]


def echantillon(voix: str) -> str:
    """« ref:<chemin>|pitch=1.4 » → chemin (la transformation d'âge est appliquée par le moteur)."""
    genre, _, valeur = voix.split("|")[0].partition(":")
    if genre != "ref":
        raise ValueError(f"Chatterbox ne sait cloner qu'un échantillon (voix reçue : {genre})")
    return valeur


def main(chemin_tache: str) -> None:
    import soundfile as sf
    import torch
    import torchaudio.functional as F
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    tache = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    dossier = Path(tache["dossier"])
    langue = langue_chatterbox(tache["langue"])
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    modele = ChatterboxMultilingualTTS.from_pretrained(device=appareil)
    print("PRET", flush=True)
    empreintes: dict[str, object] = {}

    def lire(texte: str, voix: str, force: float, cfg: float) -> np.ndarray:
        ref = echantillon(voix)
        if ref not in empreintes:  # empreinte de la voix calculée une seule fois
            modele.prepare_conditionals(ref, exaggeration=force)
            empreintes[ref] = modele.conds
        modele.conds = empreintes[ref]
        wav = modele.generate(texte, language_id=langue, exaggeration=force, cfg_weight=cfg)
        if modele.sr != TAUX:
            wav = F.resample(wav, modele.sr, TAUX)
        return rogner_silences(wav.squeeze(0).cpu().numpy().astype(np.float32))

    for seg in tache["segments"]:
        force = intensite(seg.get("temperature", 0.7))
        wav = lire(seg["traduction"], seg["voix"], force, 0.5)
        if len(wav) / TAUX / seg["disponible"] > ACCELERATION_MAX:
            # Trop long : un guidage plus faible fait parler plus vite (conseil de Resemble AI)
            essai = lire(seg["traduction"], seg["voix"], force, 0.3)
            wav = essai if len(essai) < len(wav) else wav
        sf.write(dossier / f"brut_{seg['id']:05d}.wav", wav, TAUX, subtype="PCM_16")
        print(f"FAIT {seg['id']} 1.0", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        main(sys.argv[1])
    except Exception as exc:  # message lisible pour le serveur
        print(f"ERREUR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
