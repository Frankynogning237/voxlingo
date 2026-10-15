# CONSIGNE — Lancement du projet « Doublage vidéo IA » (version locale gratuite)

> Fichier destiné à Claude Code, dans le terminal de VS Code.
> Mode d'emploi : crée un dossier vide pour le projet, place ce fichier à la racine, ouvre le dossier dans VS Code, lance `claude` dans le terminal, puis colle **une phase à la fois** (la commande du bloc « Commande à coller »). Ne passe à la phase suivante qu'après validation des critères de la phase en cours.

---

## 1. Contexte et objectif

Logiciel **local et gratuit**, pour un usage personnel :

1. l'utilisateur téléverse une vidéo, quelle que soit sa langue ;
2. le logiciel détecte la langue, transcrit, identifie chaque locuteur ;
3. il traduit dans la langue choisie en respectant le sens, les expressions, le ton et une longueur proche de l'original ;
4. il recrée **la voix propre à chaque locuteur** (clonage de voix) dans la langue cible ;
5. il cale le nouvel audio sur le timing d'origine, conserve le fond sonore et, en option, synchronise les lèvres ;
6. il propose le téléchargement de la vidéo doublée (+ sous-titres .srt).

---

## 2. Règles du projet (à respecter dans TOUTES les phases)

- **100 % local et gratuit** : aucune API payante, aucune clé payante. Uniquement des outils open source exécutés sur le PC.
- **Système** : Windows 10/11. Toutes les commandes doivent fonctionner sous Windows (PowerShell). Utiliser des chemins compatibles Windows (`pathlib`).
- **GPU** : détecter automatiquement une carte NVIDIA (CUDA). Si absente → mode CPU avec modèles plus légers. Ne jamais planter faute de GPU : afficher un avertissement clair.
- **Mémoire GPU** : charger les modèles **un par un**, les décharger après usage (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`), pour tenir dans 8 à 12 Go de VRAM.
- **Configuration centralisée** dans `backend/config.py` (modèles, langues, chemins, device), surchargeable par un fichier `.env`.
- **Interface en français**, propre et moderne.
- **Code** : TypeScript côté frontend, Python typé côté backend, fonctions courtes et commentées en français, journalisation (`logging`) de chaque étape avec sa durée.
- **Git** : initialiser un dépôt, un commit à la fin de chaque phase validée (`feat: phase X …`). `.gitignore` excluant `data/`, `models/`, `.env`, `node_modules/`, environnements Python.
- **Ne jamais committer** de vidéos, de modèles ou de fichiers lourds.
- **Mettre à jour, ne pas recréer** : à partir de la phase 2, modifier les fichiers existants au lieu de tout régénérer.
- À la fin de chaque phase : résumer ce qui a été fait, la commande pour tester, et les éventuels problèmes rencontrés.

---

## 3. Architecture cible

```
video-dubbing/
├── CONSIGNE-lancement-doublage-video.md
├── README.md
├── .gitignore
├── .env.example
├── start.ps1                     # lance backend + frontend en une commande
├── backend/
│   ├── environment.yml           # environnement conda (Python 3.11)
│   ├── requirements.txt
│   ├── main.py                   # FastAPI (http://localhost:8000)
│   ├── config.py
│   ├── jobs.py                   # gestion des tâches + statut (JSON)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── runner.py             # orchestre les étapes
│   │   ├── extract.py            # FFmpeg : extraction audio
│   │   ├── separate.py           # Demucs : voix / fond
│   │   ├── transcribe.py         # faster-whisper / WhisperX
│   │   ├── diarize.py            # pyannote.audio
│   │   ├── translate.py          # Ollama (Qwen 2.5 7B) / fallback NLLB
│   │   ├── voice.py              # XTTS v2 : clonage + synthèse
│   │   ├── align.py              # ajustement de durée (atempo)
│   │   ├── lipsync.py            # LatentSync 1.5 / MuseTalk (optionnel)
│   │   └── assemble.py           # mixage + export vidéo + .srt
│   └── tests/
├── frontend/                     # React + Vite + TypeScript + Tailwind + shadcn/ui
│   └── src/
└── data/
    └── projects/<id>/            # input.mp4, audio, segments.json, output.mp4…
```

**Fonctionnement** : le frontend envoie la vidéo à FastAPI → FastAPI crée une tâche et lance le pipeline dans un thread en arrière-plan → le frontend interroge `GET /jobs/{id}` toutes les 2 s pour afficher la progression → à la fin, lien de téléchargement.

**Pile retenue** :

| Rôle | Outil |
| --- | --- |
| Interface | React + Vite + TypeScript, Tailwind, shadcn/ui, TanStack Query |
| Serveur local | FastAPI + Uvicorn |
| Audio / vidéo | FFmpeg |
| Séparation voix / fond | Demucs (htdemucs) |
| Transcription | faster-whisper `large-v3-turbo` (GPU) / `small` (CPU), horodatage au mot via WhisperX |
| Locuteurs | pyannote.audio `speaker-diarization-3.1` |
| Traduction | Ollama + `qwen2.5:7b` (fallback : NLLB-200 distilled 600M) |
| Voix clonée | XTTS v2 (paquet `coqui-tts`, fork maintenu) |
| Synchro labiale (option) | LatentSync 1.5 (≥ 8 Go VRAM) ou MuseTalk 1.5 |

---

## 4. Prérequis à installer manuellement (avant la phase 0)

À faire toi-même, une seule fois :

1. **Git** : https://git-scm.com
2. **Node.js LTS** : https://nodejs.org
3. **Miniconda** (Python) : https://docs.conda.io/projects/miniconda
4. **FFmpeg** : `winget install Gyan.FFmpeg` dans PowerShell, puis redémarrer le terminal.
5. **Ollama** : https://ollama.com (pour la traduction locale).
6. **Pilote NVIDIA à jour** (si tu as une carte NVIDIA).
7. **Compte Hugging Face gratuit** + jeton d'accès (lecture), et accepter les conditions des modèles `pyannote/speaker-diarization-3.1` et `pyannote/segmentation-3.0` (nécessaire en phase 6).
8. **Claude Code** déjà installé et connecté.

Prévoir environ **50 Go d'espace disque** et une bonne connexion pour le téléchargement des modèles (une seule fois).

---

## PHASE 0 — Diagnostic de la machine

**Objectif** : savoir quel matériel est disponible et ce qui manque.

**Commande à coller :**

```
Lis entièrement le fichier CONSIGNE-lancement-doublage-video.md et respecte toutes ses règles (section 2). Exécute la PHASE 0 : diagnostic de ma machine sous Windows. Vérifie et rapporte dans un tableau : version de Windows, CPU, RAM totale, espace disque libre, présence et modèle de carte NVIDIA + VRAM (nvidia-smi), version du pilote et de CUDA, versions de git, node, npm, conda, python, ffmpeg, ollama. Pour chaque outil manquant, donne la commande exacte d'installation sous Windows. Conclus par le mode recommandé : "GPU local" (NVIDIA ≥ 8 Go), "GPU limité" (NVIDIA < 8 Go) ou "CPU uniquement", et dis quelles fonctionnalités seront possibles en local (notamment la synchro labiale). Ne crée encore aucun fichier de code.
```

**Validation** : tableau de diagnostic obtenu, tous les outils de la section 4 installés.

---

## PHASE 1 — Squelette du projet et environnement

**Objectif** : structure de dossiers, Git, environnement Python, serveur FastAPI minimal, frontend React minimal.

**Commande à coller :**

```
Exécute la PHASE 1 de CONSIGNE-lancement-doublage-video.md. 1) Crée l'arborescence de la section 3 (fichiers pipeline vides avec une docstring décrivant leur rôle). 2) Initialise Git avec un .gitignore adapté. 3) Crée backend/environment.yml (Python 3.11) et installe l'environnement conda "dubbing" ; installe PyTorch avec CUDA si ma machine a une carte NVIDIA (sinon version CPU), puis fastapi, uvicorn, python-multipart, pydantic-settings. 4) Crée backend/config.py qui détecte automatiquement le device (cuda/cpu) et choisit les modèles selon le mode. 5) Crée backend/main.py avec GET /health qui renvoie le device, la VRAM disponible et la version des outils. 6) Crée le frontend avec Vite + React + TypeScript + Tailwind + shadcn/ui, une page d'accueil en français qui appelle /health et affiche l'état du système. 7) Crée start.ps1 qui lance backend (port 8000) et frontend (port 5173). 8) Rédige le README.md (installation + lancement). Teste que tout démarre, puis fais le commit.
```

**Validation** : `.\start.ps1` lance les deux serveurs ; la page affiche « GPU : … » ou « CPU ».

---

## PHASE 2 — Téléversement, tâches et transcription

**Objectif** : envoyer une vidéo, extraire l'audio, détecter la langue, transcrire avec horodatage.

**Commande à coller :**

```
Exécute la PHASE 2 de CONSIGNE-lancement-doublage-video.md (mettre à jour, ne pas recréer). 1) Endpoint POST /jobs : reçoit une vidéo (mp4, mov, mkv, webm, max 2 Go, envoi par morceaux si nécessaire) + langue cible, crée data/projects/<id>/, enregistre input.<ext>, crée status.json. 2) jobs.py : exécution du pipeline dans un thread en arrière-plan, statut (en_attente, en_cours, terminé, erreur), étape courante, pourcentage, messages d'erreur lisibles. GET /jobs/{id} et GET /jobs (historique). 3) extract.py : FFmpeg extrait l'audio en WAV mono 16 kHz. 4) transcribe.py : faster-whisper (large-v3-turbo en GPU, small en CPU, compute_type adapté), détection automatique de la langue, segments avec début/fin, sauvegarde dans segments.json ; décharge le modèle après usage. 5) Frontend : page de téléversement par glisser-déposer, choix de la langue cible (fr, en, de, es, it, pt + autres langues supportées par XTTS), barre de progression par étape, affichage de la transcription. Teste avec une courte vidéo de test (génère-en une avec FFmpeg si besoin), puis commit.
```

**Validation** : une vidéo courte donne une transcription correcte et la bonne langue détectée.

---

## PHASE 3 — Traduction contextuelle locale

**Objectif** : traduire chaque segment en gardant sens, ton, expressions et une longueur proche.

**Commande à coller :**

```
Exécute la PHASE 3 de CONSIGNE-lancement-doublage-video.md. 1) Vérifie qu'Ollama tourne et télécharge le modèle qwen2.5:7b (ou qwen2.5:3b si VRAM < 8 Go / mode CPU). 2) translate.py : traduction segment par segment via l'API locale d'Ollama (http://localhost:11434), en envoyant le contexte des segments voisins. Le prompt doit exiger : fidélité au sens, conservation du registre et des expressions idiomatiques (équivalent naturel dans la langue cible), longueur la plus proche possible de l'original (objectif ±15 % en nombre de syllabes), réponse en JSON strict. Fallback automatique vers NLLB-200 distilled 600M si Ollama est indisponible. 3) Sauvegarde dans segments.json (texte original + traduction). 4) Frontend : écran d'édition côte à côte (original | traduction), chaque traduction modifiable, bouton "Valider et générer la voix". Le pipeline doit s'arrêter après la traduction et attendre cette validation (statut "a_valider"). Teste, puis commit.
```

**Validation** : traduction affichée, modifiable, et le pipeline attend la validation.

---

## PHASE 4 — Voix clonée et assemblage (premier doublage complet)

**Objectif** : générer la voix traduite avec la voix d'origine, caler les durées, exporter la vidéo.

**Commande à coller :**

```
Exécute la PHASE 4 de CONSIGNE-lancement-doublage-video.md. 1) Installe coqui-tts (fork maintenu de XTTS v2) dans l'environnement "dubbing" et télécharge le modèle xtts_v2 (accepter la licence CPML pour usage personnel non commercial). 2) voice.py : extrait de l'audio original un échantillon de référence propre de 6 à 15 s du locuteur, puis synthétise chaque segment traduit avec XTTS v2 dans la langue cible ; décharge le modèle après usage. 3) align.py : pour chaque segment, compare la durée générée à la durée d'origine ; ajuste avec FFmpeg atempo entre 0,85 et 1,25 maximum ; au-delà, signale le segment comme "trop long" dans segments.json pour retouche de la traduction. 4) assemble.py : place chaque segment à son horodatage sur une piste vide, mixe, remplace l'audio de la vidéo (la vidéo n'est pas réencodée : -c:v copy), génère aussi un .srt dans la langue cible. 5) GET /jobs/{id}/download (vidéo) et /jobs/{id}/subtitles (srt). 6) Frontend : lecteur vidéo pour comparer original et doublage, boutons de téléchargement, liste des segments signalés "trop longs" avec possibilité de modifier la traduction et de régénérer uniquement ces segments. Teste de bout en bout, puis commit.
```

**Validation** : une vidéo à un seul locuteur est doublée avec une voix proche de l'originale et téléchargeable.

---

## PHASE 5 — Conservation du fond sonore

**Objectif** : garder musique et ambiance d'origine sous la nouvelle voix.

**Commande à coller :**

```
Exécute la PHASE 5 de CONSIGNE-lancement-doublage-video.md. 1) Installe Demucs. 2) separate.py : sépare l'audio en voix et accompagnement (modèle htdemucs, mode --two-stems=vocals), sur GPU si disponible. 3) Utilise la piste "voix" isolée pour la transcription et pour l'échantillon de référence XTTS (meilleure qualité de clonage). 4) assemble.py : mixe la nouvelle voix avec la piste d'accompagnement d'origine, avec une option de volume du fond dans l'interface (0 à 100 %, défaut 100 %). 5) Mets à jour les étapes et la progression dans le frontend. Teste avec une vidéo qui contient de la musique, puis commit.
```

**Validation** : la musique de fond d'origine est conservée, la voix d'origine a disparu.

---

## PHASE 6 — Plusieurs locuteurs, une voix chacun

**Objectif** : identifier qui parle et donner à chacun sa propre voix clonée.

**Commande à coller :**

```
Exécute la PHASE 6 de CONSIGNE-lancement-doublage-video.md. 1) Installe pyannote.audio et WhisperX. Demande-moi mon jeton Hugging Face et stocke-le uniquement dans .env (HF_TOKEN), jamais dans le code ni dans Git. 2) diarize.py : diarisation avec pyannote/speaker-diarization-3.1, puis attribution de chaque segment à un locuteur (SPEAKER_00, SPEAKER_01…) ; alignement au mot avec WhisperX. 3) voice.py : un échantillon de référence par locuteur (le meilleur extrait propre de 6 à 15 s), synthèse de chaque segment avec la voix de son locuteur. 4) Frontend : liste des locuteurs détectés avec écoute de leur échantillon, possibilité de les renommer, de fusionner deux locuteurs, et de changer le locuteur d'un segment. 5) Gère les chevauchements de parole sans planter (segment attribué au locuteur majoritaire). Teste avec une vidéo à 2 personnes, puis commit.
```

**Validation** : dans une vidéo à 2 personnes, chacune garde sa propre voix dans la langue cible.

---

## PHASE 7 — Émotion et intonation

**Objectif** : retrouver le ton de l'original (joie, colère, calme…).

**Commande à coller :**

```
Exécute la PHASE 7 de CONSIGNE-lancement-doublage-video.md. 1) Pour chaque segment, analyse la prosodie de l'original : hauteur moyenne (pitch), énergie, débit (librosa ou parselmouth) et, si possible, l'émotion (emotion2vec). Enregistre ces valeurs dans segments.json. 2) voice.py : pour chaque segment, choisis comme référence XTTS un extrait du même locuteur dont l'émotion et l'énergie sont les plus proches, et ajuste les paramètres de génération (temperature, speed) en fonction du débit d'origine. 3) Ajuste le volume de chaque segment généré pour suivre l'énergie de l'original. 4) Frontend : affiche l'émotion détectée par segment et permets de la corriger. Teste, puis commit.
```

**Validation** : un passage crié ou chuchoté garde une intensité comparable une fois doublé.

---

## PHASE 8 — Synchronisation labiale (optionnelle)

**Objectif** : faire correspondre les lèvres au nouvel audio.

**Commande à coller :**

```
Exécute la PHASE 8 de CONSIGNE-lancement-doublage-video.md. Vérifie d'abord ma VRAM. Si ≥ 8 Go : installe LatentSync 1.5 (dépôt bytedance/LatentSync, checkpoints 1.5) dans un environnement conda séparé "lipsync" pour éviter les conflits de dépendances, et appelle-le depuis lipsync.py en sous-processus. Si < 8 Go ou CPU : n'installe rien en local, crée à la place un notebook Kaggle (lipsync_kaggle.ipynb) prêt à l'emploi qui prend la vidéo doublée et produit la version synchronisée, avec un guide d'utilisation dans le README. Dans les deux cas : option "Synchronisation labiale" (désactivée par défaut) dans l'interface, traitement uniquement des passages où un visage est visible de face, avertissement sur le temps de calcul. Teste sur 10 s de vidéo, puis commit.
```

**Validation** : sur un plan de face, les lèvres suivent le nouvel audio.

---

## PHASE 9 — Finitions

**Commande à coller :**

```
Exécute la PHASE 9 de CONSIGNE-lancement-doublage-video.md. 1) Page "Mes projets" : historique, reprise d'un projet, suppression (fichiers compris). 2) Page "Paramètres" : choix des modèles, mode GPU/CPU forcé, dossier de stockage, volume du fond par défaut. 3) Gestion propre des erreurs (messages clairs en français, bouton "Réessayer l'étape"). 4) Possibilité d'annuler une tâche en cours. 5) Nettoyage automatique des fichiers temporaires. 6) Tests unitaires de base (pytest) sur align.py et assemble.py. 7) Mise à jour complète du README. Commit final.
```

---

## 5. Rappels importants

- **Usage personnel uniquement** avec cette pile : XTTS v2 (licence CPML) et NLLB-200 (CC-BY-NC) interdisent l'usage commercial. Pour une version commerciale plus tard, il faudra les remplacer (ex. Fish Speech ou CosyVoice2 pour la voix).
- **Consentement** : ne doubler que des vidéos dont tu as les droits, et des personnes qui sont d'accord pour que leur voix soit clonée.
- **Sans carte NVIDIA**, tout fonctionne en CPU mais lentement (compter plusieurs minutes par minute de vidéo) ; la synchro labiale passe alors par Kaggle (environ 30 h de GPU gratuites par semaine).
- En cas d'erreur d'installation (CUDA, PyTorch, conflits de versions), demander à Claude Code : « Analyse l'erreur, explique la cause en français et corrige sans casser ce qui fonctionne ».
