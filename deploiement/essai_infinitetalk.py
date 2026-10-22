"""Essai de faisabilité : recréer un extrait d'un projet VoxLingo d'après la voix doublée (InfiniteTalk, Modal).

InfiniteTalk (MeiGen-AI, Wan 2.1 14B) refait une vidéo existante à partir d'une nouvelle voix : lèvres,
expressions, mouvements de tête et du haut du corps suivent la voix doublée ; décor, visages et éclairage
viennent de la vidéo d'origine. Cet essai mesure, sur un extrait de quelques secondes, le temps de calcul
et le rendu, avant toute intégration dans VoxLingo.

Il tourne sur Modal (carte A10G, 24 Go, en mode mémoire réduite ; L40S avec ESSAI_GPU=L40S) : le modèle demande une carte récente (bfloat16, flash-attention),
que ni ce PC ni la T4 de Kaggle n'offrent. Modèle compressé (fp8, ~25 Go téléchargés une seule fois dans un
volume Modal permanent). Coût indicatif : quelques dizaines de minutes de carte graphique, couvertes par les crédits
gratuits mensuels de Modal.

Usage, depuis la racine du projet (Modal installé et connecté : `pip install modal`, `modal token new`) :
    modal run deploiement/essai_infinitetalk.py --projet 7705009c58d3 --debut 22 --duree 8

Résultats dans data/projects/<projet>/essai_infinitetalk/ :
    recree.mp4        l'extrait recréé (480p) avec le son doublé ;
    comparaison.mp4   original | recréé, côte à côte, avec le son doublé ;
    rapport.txt       temps de calcul, réglages, fin du journal du modèle.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("voxlingo-essai-infinitetalk")
poids = modal.Volume.from_name("voxlingo-infinitetalk-poids", create_if_missing=True)
POIDS = "/poids"
CODE = "/opt/InfiniteTalk"
# Carte : A10G (24 Go) par défaut, utilisable sans moyen de paiement enregistré ; L40S ou A100 (plus rapides,
# sans mode mémoire réduite) si un moyen de paiement est enregistré sur Modal : ESSAI_GPU=L40S modal run …
GPU = os.environ.get("ESSAI_GPU", "A10G")
MEMOIRE_REDUITE = GPU not in ("L40S", "A100-80GB", "H100")
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
              "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
TORCH = ["torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(*TORCH, index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.28", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("ninja", "psutil", "packaging", "wheel", FLASH_ATTN)
    .run_commands(
        f"git clone --depth 1 https://github.com/MeiGen-AI/InfiniteTalk {CODE}",
        # Les dépendances du projet ne doivent pas remplacer la version de PyTorch choisie ci-dessus
        "printf '%s\\n' " + " ".join(TORCH) + " > /tmp/contraintes.txt",
        f"pip install -c /tmp/contraintes.txt -r {CODE}/requirements.txt librosa 'huggingface_hub[hf_transfer]'",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)


@app.function(image=image, volumes={POIDS: poids}, timeout=3 * 3600, cpu=4, memory=8192)
def preparer() -> str:
    """Télécharge, une seule fois, les modèles nécessaires au mode compressé (fp8) dans le volume."""
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        "Wan-AI/Wan2.1-I2V-14B-480P", local_dir=f"{POIDS}/Wan2.1-I2V-14B-480P",
        # Le transformeur et l'encodeur de texte complets (75 Go) sont remplacés par leur version compressée
        allow_patterns=["config.json", "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                        "google/*", "xlm-roberta-large/*"],
    )
    snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=f"{POIDS}/chinese-wav2vec2-base")
    hf_hub_download("TencentGameMate/chinese-wav2vec2-base", "model.safetensors", revision="refs/pr/1",
                    local_dir=f"{POIDS}/chinese-wav2vec2-base")
    snapshot_download(
        "MeiGen-AI/InfiniteTalk", local_dir=f"{POIDS}/InfiniteTalk",
        allow_patterns=["single/infinitetalk.safetensors", "quant_models/infinitetalk_single_fp8.*",
                        "quant_models/t5_fp8.safetensors", "quant_models/t5_map_fp8.json", "quant_models/quant.json"],
    )
    poids.commit()
    taille = sum(f.stat().st_size for f in Path(POIDS).rglob("*") if f.is_file())
    return f"{taille / 1e9:.1f} Go de modèles prêts"


@app.function(image=image, volumes={POIDS: poids}, gpu=GPU, timeout=3 * 3600, memory=65536)
def recreer(video: bytes, voix: bytes, consigne: str, etapes: int) -> dict:
    """Recrée la vidéo d'après la voix (mode vidéo vers vidéo, une personne). Renvoie la vidéo et le journal."""
    import json

    travail = Path("/tmp/essai")
    travail.mkdir(exist_ok=True)
    (travail / "origine.mp4").write_bytes(video)
    (travail / "voix.wav").write_bytes(voix)
    (travail / "entree.json").write_text(json.dumps({
        "prompt": consigne,
        "cond_video": str(travail / "origine.mp4"),
        "cond_audio": {"person1": str(travail / "voix.wav")},
    }), encoding="utf-8")
    commande = [
        "python", "generate_infinitetalk.py",
        "--ckpt_dir", f"{POIDS}/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", f"{POIDS}/chinese-wav2vec2-base",
        "--infinitetalk_dir", f"{POIDS}/InfiniteTalk/single/infinitetalk.safetensors",
        "--quant", "fp8", "--quant_dir", f"{POIDS}/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors",
        "--input_json", str(travail / "entree.json"),
        "--size", "infinitetalk-480", "--sample_steps", str(etapes),
        "--mode", "streaming", "--motion_frame", "9", "--use_teacache",
        "--save_file", str(travail / "recree"),
    ]
    if MEMOIRE_REDUITE:  # 24 Go : le transformeur est chargé morceau par morceau sur la carte
        commande += ["--num_persistent_param_in_dit", "0"]
    debut = time.time()
    resultat = subprocess.run(commande, cwd=CODE, capture_output=True, text=True)
    duree = time.time() - debut
    journal = (resultat.stdout + resultat.stderr)[-6000:]
    sortie = next(travail.glob("recree*.mp4"), None)
    return {"video": sortie.read_bytes() if sortie else None, "secondes": round(duree), "code": resultat.returncode,
            "journal": journal}


def _ffmpeg(*arguments: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *arguments], check=True)


@app.local_entrypoint()
def main(projet: str, debut: float = 22.0, duree: float = 8.0, etapes: int = 40,
         consigne: str = "A person is talking, natural facial expressions and head movements"):
    dossier = Path("data/projects") / projet
    source = next(dossier.glob("input.*"))
    sortie = dossier / "essai_infinitetalk"
    sortie.mkdir(exist_ok=True)
    extrait, voix, mixage = sortie / "origine.mp4", sortie / "voix.wav", sortie / "mixage.wav"
    # Extrait d'origine (image), voix doublée seule (pilote le mouvement), son doublé complet (résultat final)
    _ffmpeg("-ss", f"{debut}", "-t", f"{duree}", "-i", str(source), "-an", "-c:v", "libx264", "-crf", "16",
            str(extrait))
    _ffmpeg("-ss", f"{debut}", "-t", f"{duree}", "-i", str(dossier / "voix_doublee.wav"), "-ac", "1", "-ar", "16000",
            str(voix))
    _ffmpeg("-ss", f"{debut}", "-t", f"{duree}", "-i", str(dossier / "doublage.wav"), str(mixage))

    print("Préparation des modèles (la première fois : environ 25 Go à télécharger)…")
    print(preparer.remote())
    print(f"Recréation de {duree:.1f} s sur {GPU} ({etapes} étapes)…")
    r = recreer.remote(extrait.read_bytes(), voix.read_bytes(), consigne, etapes)
    rapport = [f"Projet {projet}, extrait {debut} s → {debut + duree} s", f"Étapes : {etapes}",
               f"Calcul : {r['secondes']} s ({r['secondes'] / duree:.0f} s de calcul par seconde de vidéo)",
               f"Code de sortie : {r['code']}", "", r["journal"]]
    (sortie / "rapport.txt").write_text("\n".join(rapport), encoding="utf-8")
    if not r["video"]:
        print(f"Échec de la recréation : voir {sortie / 'rapport.txt'}")
        return
    brute = sortie / "recree_sans_son.mp4"
    brute.write_bytes(r["video"])
    _ffmpeg("-i", str(brute), "-i", str(mixage), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(sortie / "recree.mp4"))
    _ffmpeg("-i", str(extrait), "-i", str(brute), "-i", str(mixage), "-filter_complex",
            "[0:v]scale=-2:480[a];[1:v]scale=-2:480[b];[a][b]hstack=inputs=2[v]", "-map", "[v]", "-map", "2:a:0",
            "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", str(sortie / "comparaison.mp4"))
    brute.unlink()
    print(f"Terminé en {r['secondes']} s de calcul : {sortie / 'comparaison.mp4'}")
