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

# 3. Voix clonée (XTTS v2) : PyTorch ≥ 2.9 lit l'audio avec torchcodec, qui a besoin
#    des bibliothèques FFmpeg partagées dans l'environnement conda
pip install "coqui-tts[codec]"
conda install -n dubbing --override-channels -c conda-forge "ffmpeg>=6,<8" -y

# 4. Frontend
cd frontend
npm install
cd ..

# 5. Configuration (facultatif)
Copy-Item .env.example .env
```

## Lancement

```powershell
.\start.ps1
```

- Interface : http://localhost:5173
- API : http://localhost:8000 (état du système : http://localhost:8000/health)

`Ctrl+C` arrête les deux serveurs. Le backend tourne dans une fenêtre réduite « VoxLingo API » ; ses journaux sont aussi écrits dans `data/logs/voxlingo.log`.

Si PowerShell refuse d'exécuter le script : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Vidéos acceptées

MP4, MOV, MKV, WEBM, AVI, M4V, WMV, FLV, TS et MPG, **sans limite de taille ni de durée** : seul l'espace disque compte (au moins 2 Go doivent rester libres). Les traitements lisent l'audio et la vidéo par morceaux, pour rester légers en mémoire même sur un film de plusieurs heures.

En mode CPU, comptez environ une minute de calcul par minute de vidéo pour la transcription et la traduction (plus pour les phases suivantes).

## Traduction

La traduction se fait phrase par phrase avec Ollama (`qwen2.5:3b` en mode CPU, `qwen2.5:7b` avec une carte NVIDIA ≥ 8 Go), en donnant au modèle les phrases voisines comme contexte et une longueur cible. Si Ollama ne répond pas, NLLB-200 prend le relais automatiquement.

Le traitement s'arrête ensuite sur l'écran **Édition de la traduction** : chaque traduction est modifiable, l'alerte « Trop long » signale les phrases que la voix ne pourra pas prononcer dans le temps d'origine (plus de 25 % de syllabes en plus), et « Valider et générer les voix » lance la suite.

## Voix doublée

Après « Valider et générer les voix », VoxLingo :

1. extrait 6 à 15 s de la voix d'origine et la clone avec **XTTS v2** (modèle de 1,8 Go téléchargé une fois dans `models/tts`) ;
2. fait lire chaque phrase traduite par cette voix (les voix générées sont gardées en cache : après une correction, seules les phrases modifiées sont refaites) ;
3. cale chaque voix sur son temps (accélération jusqu'à ×1,25, ralentissement jusqu'à ×0,85) et signale les phrases qui ne tiennent toujours pas ;
4. assemble la piste, remplace l'audio de la vidéo **sans réencoder l'image**, et produit les sous-titres `.srt`.

L'écran **Résultat** propose le comparateur avant/après, le téléchargement de la vidéo (MP4), des sous-titres (SRT) et de l'audio seul (MP3), et un retour à l'éditeur pour corriger puis régénérer.

> Avec « Conserver le fond sonore », la musique et l'ambiance d'origine (séparées par Demucs) sont remixées sous la voix doublée.

## Locuteurs et voix

Après la transcription, VoxLingo **identifie automatiquement tous les locuteurs**, quelle que soit la vidéo :

1. **pyannote.audio** (modèle gratuit `pyannote/speaker-diarization-community-1`), l'outil open source de référence,
   trouve qui parle et quand, même quand les personnes se coupent la parole ;
2. **registres de hauteur** : si une même voix joue plusieurs personnages (vidéos générées par IA, doublages),
   ses répliques sont séparées par leur hauteur de voix (un garçon à 240 Hz, une fille à 400 Hz…) ;
3. une phrase dite par plusieurs personnes est **découpée en répliques**, chacune avec son locuteur.

pyannote demande, une seule fois, un compte gratuit sur huggingface.co : accepter les conditions de
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
créer un jeton *Read* (Settings → Access Tokens) et l'écrire dans `.env` (`HF_TOKEN=…`, secret `HF_TOKEN` sur
Kaggle/Colab). Sans jeton, une analyse locale (empreintes SpeechBrain ECAPA + hauteur de voix) prend le relais.

Chaque locuteur reçoit ensuite **sa propre voix** : la sienne clonée, ou, s'il parle trop peu (moins de 3 s) pour
être cloné, une voix de la bibliothèque différente de celles des autres et proche de sa hauteur de voix.
La page **Locuteurs** permet de renommer, fusionner, corriger le locuteur d'une phrase et choisir la voix :

- **voix d'origine clonée** (par défaut) ;
- **voix de la bibliothèque** : 20 voix (4 enfants, 4 jeunes, 8 adultes, 4 seniors, femmes et hommes) ;
  « Attribuer automatiquement » donne à chaque personne une voix différente, du même sexe et du registre le plus proche ;
- **vos propres voix** : un enregistrement de 6 à 15 s ajouté dans la Bibliothèque (Paramètres → Bibliothèque de voix).

> Les voix d'enfants, de jeunes et de seniors sont obtenues en transformant des voix d'adultes XTTS (hauteur,
> résonances, tempo, léger tremblement). Pour un rendu parfaitement naturel, ajoutez un vrai enregistrement.

Après un changement de voix, cliquez sur **Générer les voix** : seules les phrases concernées sont refaites.
Une seule génération de voix tourne à la fois (8 Go de RAM ne suffisent pas pour deux) : les suivantes attendent leur tour.

## Ton des phrases

Chaque phrase d'origine est analysée (énergie, hauteur, variation, débit) et reçoit un ton : énergique, intense,
calme, doux ou neutre, par rapport à la façon habituelle de parler de la même personne. Le ton règle
l'expressivité de la voix doublée, et son volume suit celui de la phrase d'origine (une phrase criée reste forte).
C'est une analyse acoustique, pas une reconnaissance d'émotions : le ton se corrige d'un clic dans l'éditeur.

## Synchronisation labiale (Kaggle)

Sans carte NVIDIA, la synchronisation des lèvres (LatentSync 1.5) se fait sur le GPU gratuit de Kaggle :

1. écran **Résultat** → *Télécharger le paquet Kaggle* (vidéo, voix doublée, carnet `kaggle/lipsync_kaggle.ipynb`, guide) ;
2. sur kaggle.com : importer le carnet et le paquet, choisir un GPU, *Run All* ;
3. écran Résultat → *Importer la vidéo synchronisée* : elle devient la vidéo téléchargée.

## En ligne, gratuitement (Vercel + Kaggle/Colab)

Pour un PC peu puissant : l'**interface** est publiée sur Vercel (gratuit), le **moteur** tourne sur le GPU
gratuit de Kaggle (≈ 30 h/semaine) ou de Colab, joignable grâce à un tunnel Cloudflare gratuit. Coût : 0 F.

**Interface sur Vercel (une seule fois)**
1. vercel.com → *Sign up with GitHub* (offre Hobby) → *Add New… → Project* → importer le dépôt `voxlingo` ;
2. *Root Directory* : `frontend` (Vite est détecté, `frontend/vercel.json` fait le reste) → *Deploy* ;
3. noter l'adresse obtenue (par exemple `https://voxlingo.vercel.app`). Chaque `git push` republie l'interface.

**Moteur sur Kaggle (à chaque séance)**
1. kaggle.com → *Create → New Notebook → File → Import Notebook* → `kaggle/moteur_voxlingo.ipynb` ;
2. une seule fois : vérifier son téléphone (*Settings* du compte), puis dans le carnet *GPU T4 x2*, *Internet On*,
   et le secret `GITHUB_TOKEN` (jeton GitHub *fine-grained*, lecture seule du contenu de `voxlingo`) ;
3. renseigner `ADRESSE_VERCEL` dans la première cellule → *Run All* (≈ 10 min) ;
4. ouvrir le lien affiché : l'interface Vercel se connecte toute seule au moteur (adresse et clé mémorisées).

Le même carnet fonctionne sur Colab (*Fichier → Importer*, secret dans l'icône 🔑). Le moteur exige une **clé
d'accès** (`CLE_ACCES`) et n'accepte que les sites `localhost` et `*.vercel.app` (`CORS_ORIGINES`, `CORS_REGEX`).
Les vidéos partent en morceaux de 50 Mo (le tunnel refuse les envois de plus de 100 Mo), donc sans limite de taille.
Le voyant d'état en haut à droite ouvre la fenêtre **Connexion au moteur** (bouton *Moteur de ce PC* pour revenir).

À savoir : l'adresse du moteur change à chaque séance, et les fichiers de Kaggle/Colab sont effacés à la fin
(12 h max) — téléchargez vos vidéos doublées avant. XTTS v2 (CPML) et NLLB : usage personnel non commercial.

## Tests

```powershell
cd backend
python -m pytest
```

## Développement

`start.ps1` lance le serveur **sans rechargement automatique** : un redémarrage interromprait les traitements en
cours. Pour développer : `python -m uvicorn main:app --reload` dans `backend/`.

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
- [x] Phase 3 — Traduction contextuelle locale
- [x] Phase 4 — Voix clonée et assemblage
- [x] Phase 5 — Conservation du fond sonore
- [x] Phase 6 — Plusieurs locuteurs
- [x] Phase 7 — Émotion et intonation
- [x] Phase 8 — Synchronisation labiale
- [x] Phase 9 — Finitions
