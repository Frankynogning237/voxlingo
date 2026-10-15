# VoxLingo — doublage vidéo IA, local et gratuit

Logiciel de bureau qui double une vidéo dans une autre langue en gardant la voix de chaque personne :
transcription, identification des locuteurs, traduction, voix clonée, calage sur le timing d'origine,
conservation du fond sonore et, en option, synchronisation labiale.

Tout tourne sur le PC, avec des outils open source. Les étapes de développement sont décrites dans
[CONSIGNE-lancement-doublage-video.md](CONSIGNE-lancement-doublage-video.md).

> **Usage personnel uniquement** : XTTS v2 (licence CPML) et NLLB-200 (CC-BY-NC) interdisent l'usage commercial.
> Ne doublez que des vidéos dont vous avez les droits, avec l'accord des personnes dont la voix est clonée.

## Prérequis

| Outil | Installation (PowerShell) |
| --- | --- |
| Git | `winget install Git.Git` |
| Node.js LTS | `winget install OpenJS.NodeJS.LTS` |
| Miniconda | `winget install Anaconda.Miniconda3` |
| FFmpeg | `winget install Gyan.FFmpeg` |
| Ollama | `winget install Ollama.Ollama` |

Redémarrez le terminal après l'installation. Prévoyez environ 50 Go d'espace disque pour les modèles.

## Installation

Depuis la racine du projet :

```powershell
# 1. Environnement Python "dubbing" (Python 3.11)
conda create -n dubbing --override-channels -c conda-forge python=3.11 pip -y
conda activate dubbing
pip install -r backend/requirements.txt

# 2. PyTorch : version CPU (sans carte NVIDIA)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# ...ou version CUDA (avec carte NVIDIA)
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Frontend
cd frontend
npm install
cd ..

# 4. Configuration (facultatif)
Copy-Item .env.example .env
```

## Lancement

```powershell
.\start.ps1
```

- Interface : http://localhost:5173
- API : http://localhost:8000 (état du système : http://localhost:8000/health)

`Ctrl+C` arrête les deux serveurs.

Si PowerShell refuse d'exécuter le script : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Modes de fonctionnement

Le backend détecte le matériel au démarrage (`backend/config.py`) :

| Mode | Matériel | Transcription | Traduction | Synchro labiale |
| --- | --- | --- | --- | --- |
| GPU local | NVIDIA ≥ 8 Go | `large-v3-turbo` | `qwen2.5:7b` | en local |
| GPU limité | NVIDIA < 8 Go | `large-v3-turbo` | `qwen2.5:3b` | via Kaggle |
| CPU uniquement | sans NVIDIA | `small` | `qwen2.5:3b` | via Kaggle |

Les modèles peuvent être forcés dans `.env` (voir `.env.example`).

## Structure

```
backend/   API FastAPI, configuration, pipeline de traitement (une étape par fichier)
frontend/  Interface React + Vite + TypeScript + Tailwind + shadcn/ui
data/      Projets et fichiers générés (ignoré par Git)
models/    Modèles téléchargés (ignoré par Git)
```

## Avancement

- [x] Phase 0 — Diagnostic de la machine
- [x] Phase 1 — Squelette du projet et environnement
- [x] Phase 2 — Téléversement, tâches et transcription
- [ ] Phase 3 — Traduction contextuelle locale
- [ ] Phase 4 — Voix clonée et assemblage
- [ ] Phase 5 — Conservation du fond sonore
- [ ] Phase 6 — Plusieurs locuteurs
- [ ] Phase 7 — Émotion et intonation
- [ ] Phase 8 — Synchronisation labiale
- [ ] Phase 9 — Finitions
