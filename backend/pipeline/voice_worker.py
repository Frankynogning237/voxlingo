"""Processus séparé de synthèse XTTS v2, lancé par pipeline/voice.py.

XTTS demande 3 à 4 Go de mémoire. L'exécuter dans son propre processus protège le
serveur : en cas de manque de mémoire, seul ce processus s'arrête, l'interface
reste disponible et affiche un message clair.

Entrée : un fichier JSON {"segments": [{"id", "traduction", "vitesse", "disponible", "voix"}],
"langue", "dossier"}, où « voix » vaut « ref:<chemin WAV> » (voix clonée à partir d'un
échantillon) ou « xtts:<nom> » (voix intégrée à XTTS v2).
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
    print("PRET", flush=True)
    conditionnements: dict[str, tuple] = {}

    def conditionnement(voix: str, intonation: str | None = None) -> tuple:
        """Empreinte de la voix, calculée une seule fois par voix.

        Avec `intonation` (extrait de la phrase d'origine, voix clonées) : le timbre reste celui de
        l'échantillon de référence, mais le style de diction (latents GPT) s'inspire de la phrase
        d'origine, pour en garder l'émotion (colère, rire, chuchotement…)."""
        base = voix.split("|")[0]  # la transformation d'âge est appliquée après synthèse
        if intonation:
            cle = f"{voix}#{intonation}"
            if cle not in conditionnements:
                _, empreinte = conditionnement(voix)
                latent, _ = modele.get_conditioning_latents(audio_path=[intonation, base.partition(":")[2]])
                conditionnements[cle] = (latent, empreinte)
            return conditionnements[cle]
        if voix not in conditionnements:
            genre, _, valeur = base.partition(":")
            if genre == "xtts":
                donnees = modele.speaker_manager.speakers[valeur]
                conditionnements[voix] = (donnees["gpt_cond_latent"], donnees["speaker_embedding"])
            else:
                conditionnements[voix] = modele.get_conditioning_latents(audio_path=[valeur])
        return conditionnements[voix]

    def lire(texte: str, voix: str, vitesse: float, temperature: float = 0.7,
             intonation: str | None = None) -> np.ndarray:
        latent, empreinte = conditionnement(voix, intonation)
        sortie = modele.inference(
            texte, tache["langue"], latent, empreinte,
            temperature=temperature, speed=vitesse, enable_text_splitting=True,
        )
        return _rogner_silences(np.asarray(sortie["wav"], dtype=np.float32))

    for seg in tache["segments"]:
        vitesse = seg["vitesse"]
        temperature = seg.get("temperature", 0.7)  # expressivité selon le ton de la phrase d'origine
        wav = lire(seg["traduction"], seg["voix"], vitesse, temperature, seg.get("intonation"))
        depassement = len(wav) / TAUX / seg["disponible"]
        if depassement > ACCELERATION_MAX and vitesse < VITESSE_XTTS_MAX:
            # Encore trop long : nouvel essai plus rapide, avec une petite marge
            vitesse = round(min(VITESSE_XTTS_MAX, vitesse * depassement / (ACCELERATION_MAX * 0.95)), 2)
            wav = lire(seg["traduction"], seg["voix"], vitesse, temperature, seg.get("intonation"))
        sf.write(fichier_brut(dossier, seg["id"]), wav, TAUX, subtype="PCM_16")
        print(f"FAIT {seg['id']} {vitesse}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        main(sys.argv[1])
    except Exception as exc:  # message lisible pour le serveur
        print(f"ERREUR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
