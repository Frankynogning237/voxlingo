"""Sépare voix et fond avec un modèle RoFormer (appelé par pipeline/separate.py, processus séparé).

Usage : python -m pipeline.roformer_worker <audio.wav> <dossier_sortie> <dossier_modeles> <modele>
Écrit vocals.wav et no_vocals.wav dans le dossier de sortie. L'audio est traité par morceaux
(`chunk_duration`) : mémoire bornée quelle que soit la durée de la vidéo.
"""

import logging
import sys
from pathlib import Path

MORCEAU_S = 300  # 5 min par morceau


def main(arguments: list[str]) -> int:
    source, sortie, modeles, modele = arguments
    from audio_separator.separator import Separator

    separateur = Separator(
        log_level=logging.WARNING, model_file_dir=modeles, output_dir=sortie,
        sample_rate=44100, chunk_duration=MORCEAU_S,
    )
    separateur.load_model(model_filename=modele)
    fichiers = separateur.separate(source, custom_output_names={"Vocals": "vocals", "Instrumental": "no_vocals"})
    noms = {Path(f).name for f in fichiers}
    if not {"vocals.wav", "no_vocals.wav"} <= noms:
        print("Pistes inattendues :", sorted(noms), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
