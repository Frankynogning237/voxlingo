"""Processus séparé de synthèse XTTS v2, lancé par pipeline/voice.py.

XTTS demande 3 à 4 Go de mémoire. L'exécuter dans son propre processus protège le
serveur : en cas de manque de mémoire, seul ce processus s'arrête, l'interface
reste disponible et affiche un message clair.

Entrée : un fichier JSON {"segments": [{"id", "traduction", "vitesse", "disponible"}],
"langue", "reference", "dossier"}.
Sortie standard, une ligne par événement :
  PRET                      modèle chargé
  FAIT <id> <vitesse>       voix du segment écrite dans <dossier>/brut_<id>.wav

Si une voix dépasse encore son temps disponible au-delà de ce que le calage (atempo ×1,25)
peut rattraper, elle est régénérée une fois, plus vite (jusqu'à VITESSE_XTTS_MAX).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np


def main(chemin_tache: str) -> None:
    import soundfile as sf

    from config import get_runtime, get_settings
    from pipeline.segments import ACCELERATION_MAX, VITESSE_XTTS_MAX
    from pipeline.voice import TAUX, _rogner_silences, fichier_brut  # configure aussi TTS_HOME et les DLL

    tache = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    dossier = Path(tache["dossier"])
    runtime = get_runtime()

    from TTS.api import TTS

    tts = TTS(get_settings().xtts_model, progress_bar=False).to(runtime.device)
    modele = tts.synthesizer.tts_model
    latent, empreinte = modele.get_conditioning_latents(audio_path=[tache["reference"]])
    print("PRET", flush=True)

    def lire(texte: str, vitesse: float) -> np.ndarray:
        sortie = modele.inference(
            texte, tache["langue"], latent, empreinte,
            temperature=0.7, speed=vitesse, enable_text_splitting=True,
        )
        return _rogner_silences(np.asarray(sortie["wav"], dtype=np.float32))

    for seg in tache["segments"]:
        vitesse = seg["vitesse"]
        wav = lire(seg["traduction"], vitesse)
        depassement = len(wav) / TAUX / seg["disponible"]
        if depassement > ACCELERATION_MAX and vitesse < VITESSE_XTTS_MAX:
            # Encore trop long : nouvel essai plus rapide, avec une petite marge
            vitesse = round(min(VITESSE_XTTS_MAX, vitesse * depassement / (ACCELERATION_MAX * 0.95)), 2)
            wav = lire(seg["traduction"], vitesse)
        sf.write(fichier_brut(dossier, seg["id"]), wav, TAUX, subtype="PCM_16")
        print(f"FAIT {seg['id']} {vitesse}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        main(sys.argv[1])
    except Exception as exc:  # message lisible pour le serveur
        print(f"ERREUR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
