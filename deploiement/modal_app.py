"""Moteur VoxLingo sur Modal (GPU T4 à la demande, ≈ 30 $ de crédits gratuits par mois).

Déploiement, depuis la racine du dépôt (une fois, puis après chaque mise à jour du code) :
    pip install modal
    modal token new                                    # connexion au compte Modal (navigateur)
    modal secret create voxlingo CLE_ACCES=<votre mot de passe> HF_TOKEN=hf_…   # les deux facultatifs
    modal deploy deploiement/modal_app.py
L'adresse affichée (https://<compte>--voxlingo-moteur-moteur-web.modal.run) se colle dans VoxLingo,
fenêtre « Connexion au moteur », avec la clé : CLE_ACCES, ou, sans elle, la clé tirée au hasard au premier
démarrage et écrite dans les journaux de l'application (modal.com → Apps → voxlingo-moteur → Logs).

Fonctionnement :
- un seul conteneur GPU (T4), allumé à la première requête (≈ 1 à 2 min) et tant que l'interface
  est ouverte : elle interroge le moteur toutes les quelques secondes. Il s'éteint 20 min après la
  dernière requête, et ne coûte rien éteint. Fermer l'interface pendant un doublage l'interrompt ;
- projets, modèles téléchargés (Whisper, XTTS, Ollama…) et bibliothèque de voix sont sur un volume
  Modal permanent : rien n'est perdu d'une séance à l'autre ;
- Chatterbox et MuseTalk sont installés dans l'image, une fois pour toutes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

RACINE = Path(__file__).resolve().parent.parent
STOCKAGE = "/stockage"  # volume permanent
CHATTERBOX = "/opt/chatterbox"
MUSETALK = "/opt/musetalk"
FACULTATIFS = ("pyannote.audio", "mediapipe", "onnxruntime", "audio-separator", "funasr")

volume = modal.Volume.from_name("voxlingo-stockage", create_if_missing=True)

_requirements = [l for l in (RACINE / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith(("#", *FACULTATIFS))]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    # build-essential et clang : certains paquets se compilent à l'installation (diffq, pour RoFormer)
    .apt_install("ffmpeg", "git", "curl", "wget", "zstd", "lshw", "rubberband-cli", "build-essential", "clang")
    .pip_install(*(l.split("#")[0].strip() for l in _requirements))
    # Outils facultatifs : le moteur s'en passe si l'un d'eux ne s'installe pas
    .run_commands(*(f"pip install -q '{outil}' || echo '{outil} non installé'" for outil in ("torchcodec",) + FACULTATIFS))
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .run_commands(
        f"pip install -q uv && uv venv -q -p 3.11 {CHATTERBOX}"
        f" && uv pip install -q -p {CHATTERBOX}/bin/python chatterbox-tts 'setuptools<81' && touch {CHATTERBOX}/.pret"
    )
    .add_local_file(RACINE / "kaggle" / "installer_musetalk.sh", "/opt/installer_musetalk.sh", copy=True)
    .run_commands(f"MUSETALK_RACINE={MUSETALK} MUSETALK_SANS_GPU=1 bash /opt/installer_musetalk.sh")
    .env({
        "DATA_DIR": f"{STOCKAGE}/donnees", "MODELS_DIR": f"{STOCKAGE}/modeles", "OLLAMA_MODELS": f"{STOCKAGE}/ollama",
        "OLLAMA_MODEL": "qwen2.5:14b", "MODELE_VISION": "qwen2.5vl:7b", "COQUI_TOS_AGREED": "1",
        "MPLBACKEND": "Agg", "CHATTERBOX_PYTHON": f"{CHATTERBOX}/bin/python",
        "MUSETALK_PYTHON": f"{MUSETALK}/env/bin/python", "MUSETALK_DIR": f"{MUSETALK}/code",
    })
    .add_local_dir(RACINE / "backend", "/app/backend", ignore=["**/__pycache__", "tests", "**/*.pyc"])
)

app = modal.App("voxlingo-moteur", image=image)


@app.cls(
    gpu="T4",
    volumes={STOCKAGE: volume},
    secrets=[modal.Secret.from_name("voxlingo")],
    max_containers=1,  # un seul moteur : les doublages en cours vivent dans ce conteneur
    scaledown_window=20 * 60,  # éteint 20 min après la dernière requête
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=100)
class Moteur:
    @modal.enter()
    def demarrer(self) -> None:
        if not os.environ.get("CLE_ACCES"):
            # Pas de clé dans le secret : une clé tirée au hasard une fois, gardée sur le volume
            import secrets

            fichier = Path(os.environ["DATA_DIR"]) / ".cle"
            if not fichier.is_file():
                fichier.parent.mkdir(parents=True, exist_ok=True)
                fichier.write_text(secrets.token_urlsafe(18), encoding="utf-8")
                volume.commit()
            os.environ["CLE_ACCES"] = fichier.read_text(encoding="utf-8").strip()
            print(f"Clé d'accès du moteur (à coller dans VoxLingo) : {os.environ['CLE_ACCES']}", flush=True)
        Path(os.environ["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)
        self.ollama = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @modal.asgi_app()
    def web(self):
        os.chdir("/app/backend")
        sys.path.insert(0, "/app/backend")
        from main import app as api  # FastAPI : routes, clé d'accès et CORS (*.vercel.app) déjà en place

        return api

    @modal.exit()
    def arreter(self) -> None:
        self.ollama.terminate()
        volume.commit()  # dernières écritures (projets, voix) gardées sur le volume
