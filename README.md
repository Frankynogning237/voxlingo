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

**Format conservé à l'identique** : la vidéo doublée garde le conteneur (MP4, MOV, MKV, WEBM, AVI, WMV…), le codec,
la résolution, la cadence et l'orientation de la vidéo fournie ; le son doublé reprend le codec, la fréquence, le nombre
de canaux et le débit du son d'origine. L'image est copiée sans réencodage ; elle n'est réencodée (dans le même codec,
à la même taille et cadence) que pour « Raccourcir la vidéo » et pour remettre au format la vidéo synchronisée sur Kaggle.
Si le navigateur ne sait pas lire ce format, le site affiche une copie MP4 pour l'aperçu seulement.

**Raccourcir la vidéo** : sous l'aperçu, ce bouton permet de ne garder qu'un passage (curseur à deux poignées,
ou « Début ici » / « Fin ici » pendant la lecture). Seul ce passage est doublé : la vidéo est coupée
au début du traitement, à l'image près.

## Traduction

La traduction se fait phrase par phrase avec Ollama (`qwen2.5:3b` en mode CPU, `qwen2.5:7b` avec une carte NVIDIA ≥ 8 Go), en donnant au modèle les phrases voisines comme contexte et une longueur cible. Si Ollama ne répond pas, NLLB-200 prend le relais automatiquement.

Dans **Paramètres → Modèle de traduction**, on peut choisir **Qwen3 14B** (licence Apache 2.0), le plus fin sur les dialogues et les nuances, conseillé sur GPU (Kaggle). Un modèle choisi mais absent d'Ollama est téléchargé automatiquement au premier usage (≈ 9 Go pour un 14B) ; le mode « réflexion » de Qwen3 est désactivé pour traduire vite.

Le traitement s'arrête ensuite sur l'écran **Édition de la traduction** : chaque traduction est modifiable, l'alerte « Trop long » signale les phrases que la voix ne pourra pas prononcer dans le temps d'origine (plus de 25 % de syllabes en plus), et « Valider et générer les voix » lance la suite.

## Analyse de la scène (regarder et écouter avant de traduire)

Avant de traduire, l'étape **Analyse de la scène** comprend la vidéo pour que la traduction reste dans la scène plutôt que mot à mot :

- **émotion de chaque phrase** d'après la voix (emotion2vec+, paquet `funasr` : joie, colère, tristesse, peur, surprise, dégoût, neutre), la **manière de dire** comparée aux habitudes du même locuteur (crie, chuchote, rapide, monocorde…) et l'**expression du visage** de celui qui parle (sourit, fronce les sourcils…) ;
- **description des scènes** : un modèle de vision (`qwen2.5vl:7b` via Ollama, sur GPU seulement ; réglage `MODELE_VISION`, `aucun` pour le désactiver) décrit une image de chaque plan ;
- **fiche de la vidéo** rédigée par l'IA : genre, résumé, registre de langue, ton, personnages et relations.

Chaque phrase est ensuite traduite avec la fiche, la scène en cours, qui parle et avec quelle émotion ; l'émotion règle aussi l'expressivité de la voix doublée. Dans l'éditeur, l'émotion de chaque phrase se corrige d'un clic, et le bouton **Fiche** permet de relire et corriger la fiche, de relancer l'analyse ou de tout retraduire avec la fiche corrigée.

## Voix doublée

Après « Valider et générer les voix », VoxLingo :

1. extrait 6 à 15 s de la voix d'origine et la clone avec **Chatterbox** ou **XTTS v2** (voir ci-dessous) ;
2. fait lire chaque phrase traduite par cette voix (les voix générées sont gardées en cache : après une correction, seules les phrases modifiées sont refaites) ;
3. cale chaque voix sur son temps (accélération jusqu'à ×1,25, ralentissement jusqu'à ×0,85) et signale les phrases qui ne tiennent toujours pas ;
4. assemble la piste, remplace l'audio de la vidéo **sans réencoder l'image**, et produit les sous-titres `.srt`.

L'écran **Résultat** propose le comparateur avant/après, le téléchargement de la vidéo (MP4), des sous-titres (SRT) et de l'audio seul (MP3), et un retour à l'éditeur pour corriger puis régénérer.

**Fidélité à chaque voix d'origine** (automatique) :
- **volume au décibel près** : pour chaque phrase, le niveau de la voix d'origine (piste voix isolée, silences
  exclus) est mesuré et la voix doublée y est calée exactement (`pipeline/niveau.py`) : une phrase criée reste
  criée, un chuchotement reste chuchoté, et l'équilibre voix / musique de la vidéo est conservé ;
- **timbre** : c'est le clonage lui-même qui le reproduit, à partir d'un échantillon de la voix d'origine. Une
  égalisation supplémentaire vers la piste voix isolée existe (±3 dB, `pipeline/timbre_eq.py`) mais elle est
  **désactivée** par défaut (`EGALISATION_TIMBRE=true` dans `.env` pour l'essayer) : la piste isolée garde la
  réverbération et perd des aigus à la séparation, et la corriger dans ce sens éloignait les voix des originales ;
- **échantillon de clonage** (≈ 12 s par locuteur) : chaque phrase candidate reçoit une empreinte vocale
  (SpeechBrain ECAPA, le modèle de l'identification des locuteurs), comparée à la voix type du locuteur ; les
  extraits qui ne lui ressemblent pas (phrase attribuée à tort, voix superposée, musique, cri saturé) sont
  écartés, et le plus représentatif passe en tête (Chatterbox n'écoute que le début de l'échantillon). Les
  phrases où un autre locuteur parle ne servent que faute de mieux ; les extraits sont gardés tels quels (un
  seul gain global, jamais d'écrêtage) — `pipeline/ressemblance.py` ;
- **vérification de chaque voix clonée** : après la synthèse, la ressemblance de chaque phrase avec
  l'échantillon est mesurée. Une phrase nettement moins ressemblante que les autres du même locuteur (timbre
  qui a dérivé) est refaite une fois en mode stable (diction plus sage, sans l'intonation importée), et la
  meilleure des deux versions est gardée (au plus 30 % des phrases d'un locuteur). La mesure est enregistrée
  dans chaque phrase (`ressemblance`, de 0 à 1).

**Traduction** : le traducteur connaît le genre de chaque locuteur quand sa hauteur de voix est nette (accords
corrects : « je suis prête »), puis une **relecture du dialogue entier** corrige les incohérences (noms,
tutoiement/vouvoiement, accords, tournures peu naturelles) sans changer la longueur. Elle rallonge la traduction
d'environ 50 % ; `RELECTURE_TRADUCTION=false` dans `.env` la désactive.

**Moteur de voix** (*Paramètres → Modèle de voix*) :
- **Chatterbox Multilingual** (Resemble AI, licence MIT, 23 langues dont le français) : plus expressif, son intensité
  (`exaggeration`) suit le ton et l'émotion de chaque phrase. Il exige PyTorch 2.6 : il vit dans son propre
  environnement Python (`CHATTERBOX_PYTHON`), que le carnet Kaggle installe en arrière-plan (≈ 5 min ; XTTS est
  utilisé en attendant). Les voix intégrées de la bibliothèque sont clonées à partir de leur aperçu.
- **XTTS v2** (licence CPML, usage personnel) : toutes les langues proposées (dont tchèque et hongrois), seul moteur
  si Chatterbox n'est pas installé, et relais automatique si Chatterbox échoue.

« Automatique » (par défaut) prend Chatterbox dès qu'il est prêt et que la langue est couverte. Changer de moteur
régénère les voix du projet à la prochaine génération.

> Avec « Conserver le fond sonore », la musique et l'ambiance d'origine sont séparées de la voix puis remixées sous
> la voix doublée. La séparation utilise **BS-RoFormer** (paquet facultatif `audio-separator`, modèle de 640 Mo
> téléchargé une fois dans `models/audio-separator`), nettement plus propre que Demucs : moins d'écho de la voix
> d'origine dans le fond et un échantillon de clonage plus net. Sans lui (ou s'il échoue), **Demucs** prend le relais.

## Locuteurs et voix

Après la transcription, VoxLingo **identifie automatiquement tous les locuteurs**, quelle que soit la vidéo :

1. **pyannote.audio** (modèle gratuit `pyannote/speaker-diarization-community-1`), l'outil open source de référence,
   trouve qui parle et quand, même quand les personnes se coupent la parole ;
2. **registres de hauteur** : si une même voix joue plusieurs personnages (vidéos générées par IA, doublages),
   ses répliques sont séparées par leur hauteur de voix (un garçon à 240 Hz, une fille à 400 Hz…) ;
3. **lecture des scènes** : VoxLingo regarde aussi la vidéo. Il repère les visages (MediaPipe), voit quelle
   bouche bouge pendant chaque réplique et reconnaît chaque personnage d'un plan à l'autre (empreinte du
   visage ArcFace, apparence générale DINOv2, couleurs des cheveux et de la tenue). Quand on voit qui parle,
   l'image l'emporte ; sinon la réplique revient au personnage de la scène qui a cette voix. La page
   Locuteurs affiche le visage de chaque personnage (`visages/perso_N.jpg` du projet) ;
4. une phrase dite par plusieurs personnes est **découpée en répliques**, chacune avec son locuteur.

Les modèles de vision (~110 Mo, usage personnel non commercial pour ArcFace) se téléchargent au premier usage.
Sur un processeur, la lecture des scènes prend environ 0,8 fois la durée de la vidéo.

pyannote demande, une seule fois, un compte gratuit sur huggingface.co : accepter les conditions de
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
créer un jeton *Read* (Settings → Access Tokens) et l'écrire dans `.env` (`HF_TOKEN=…`, secret `HF_TOKEN` sur
Kaggle/Colab). Sans jeton, une analyse locale (empreintes SpeechBrain ECAPA + hauteur de voix) prend le relais.

Chaque locuteur reçoit ensuite **sa propre voix** : la sienne clonée, ou, s'il parle trop peu (moins de 3 s) pour
être cloné, une voix de la bibliothèque différente de celles des autres et proche de sa hauteur de voix.
La page **Locuteurs** permet de renommer, fusionner, corriger le locuteur d'une phrase et choisir la voix :

- **voix d'origine clonée** (par défaut) ;
- **voix de la bibliothèque** : 51 voix (6 enfants, 7 jeunes, 32 adultes, 6 seniors, femmes et hommes), dont
  6 variantes de style (grave, claire, chaude, douce, radio, téléphone) ;
  « Attribuer automatiquement » donne à chaque personne une voix différente, du même sexe et du registre le plus proche ;
- **vos propres voix** : un enregistrement de 6 à 15 s ajouté dans la Bibliothèque (Paramètres → Bibliothèque de voix).

> Les voix d'enfants, de jeunes et de seniors sont obtenues en transformant des voix d'adultes XTTS (hauteur,
> résonances, tempo, léger tremblement). Pour un rendu parfaitement naturel, ajoutez un vrai enregistrement.

**Timbre de la voix** (page Locuteurs, sous le choix de la voix) : 13 variantes en un clic (plus grave, plus aiguë,
chaleureuse, claire, douce, radio, robuste, menue, enfantine, âgée, dans une pièce, téléphone) et des réglages fins :
hauteur (demi-tons), timbre (résonances, sombre ↔ clair, sans changer la hauteur), graves, aigus, espace,
tremblement, effet téléphone. « Écouter ce réglage » le fait entendre aussitôt sur la voix choisie.

Après un changement de voix ou de timbre, cliquez sur **Générer les voix** : seules les phrases concernées sont refaites.
Une seule génération de voix tourne à la fois (8 Go de RAM ne suffisent pas pour deux) : les suivantes attendent leur tour.

## Ton des phrases

Chaque phrase d'origine est analysée (énergie, hauteur, variation, débit) et reçoit un ton : énergique, intense,
calme, doux ou neutre, par rapport à la façon habituelle de parler de la même personne. Le ton règle
l'expressivité de la voix doublée, et son volume suit celui de la phrase d'origine (une phrase criée reste forte).
C'est une analyse acoustique, pas une reconnaissance d'émotions : le ton se corrige d'un clic dans l'éditeur.

## Synchronisation labiale

**Par le moteur (GPU Kaggle/Colab, automatique).** Quand le moteur tourne sur le carnet Kaggle/Colab, il
synchronise lui-même les lèvres avec **MuseTalk 1.5** (Tencent Music, licence MIT, poids utilisables librement) :
cochez *Synchronisation labiale* au lancement du doublage, ou cliquez sur **Synchroniser les lèvres avec le
moteur** dans l'écran Résultat. Le moteur ne confie à MuseTalk que des morceaux de 15 s au plus, là où une voix
doublée parle ; le reste de l'image est gardé tel quel, puis tout est recollé image par image (aucun décalage)
et remis au format de la vidéo fournie. Un passage sans visage de face garde l'image d'origine ; si la
synchronisation échoue, le doublage reste terminé et la méthode manuelle ci-dessous reste proposée.

MuseTalk exige Python 3.10 et PyTorch 2.0 : le carnet l'installe en arrière-plan dans son propre environnement
(`kaggle/installer_musetalk.sh`, ≈ 10 min, réglages `MUSETALK_PYTHON` et `MUSETALK_DIR`). Pour l'essayer sur un
vrai visage depuis le carnet : `exec(open(CODE / "kaggle" / "essai_synchro.py", encoding="utf-8").read())`.

**Sans GPU (carnet Kaggle manuel).** La synchronisation des lèvres (LatentSync 1.5) se fait sur le GPU gratuit
de Kaggle :

1. écran **Résultat** → *Télécharger le paquet Kaggle* (vidéo, voix doublée, carnet `kaggle/lipsync_kaggle.ipynb`, guide) ;
2. sur kaggle.com : importer le carnet et le paquet, choisir un GPU, *Run All* ;
3. écran Résultat → *Importer la vidéo synchronisée* : elle devient la vidéo téléchargée.

## En ligne, gratuitement (Vercel + Kaggle/Colab)

Pour un PC peu puissant : l'**interface** est publiée sur Vercel (gratuit), le **moteur** tourne sur le GPU
gratuit de Kaggle (≈ 30 h/semaine) ou de Colab, joignable grâce à un tunnel Cloudflare gratuit. Coût : 0 F.

**Interface sur Vercel (une seule fois)**
1. vercel.com → *Sign up with GitHub* (offre Hobby) → *Add New… → Project* → importer le dépôt `voxlingo` ;
2. *Root Directory* : `frontend` (Vite est détecté, `frontend/vercel.json` fait le reste) → *Deploy* ;
3. noter l'adresse obtenue. **Déjà fait : l'interface est en ligne sur https://voxlingo-gray.vercel.app**
   (projet `voxlingo` de l'espace gratuit *stupora*). Pour la republier après une modification :
   `cd frontend` puis `npx vercel deploy --prod`.

**Moteur sur Kaggle**
1. une seule fois : kaggle.com → *Create → New Notebook → File → Import Notebook* → `kaggle/moteur_voxlingo.ipynb`.
   Le carnet n'est qu'un lanceur : il récupère la dernière version du code à chaque *Run All* et exécute
   `kaggle/demarrer.py` (logique dans `deploiement/lancer_moteur.py`) ; **inutile de le réimporter** après une
   mise à jour ;
2. une seule fois : vérifier son téléphone (*Settings* du compte), puis dans le carnet *GPU T4 x2*, *Internet On*,
   et le secret `GITHUB_TOKEN` (jeton GitHub *fine-grained*, lecture seule du contenu de `voxlingo`) ;
   conseillé : le secret `HF_TOKEN_ECRITURE` pour la sauvegarde automatique (voir plus bas) ;
3. à chaque séance : *Run All* (≈ 10 min ; `ADRESSE_VERCEL` et les modèles se règlent dans la première cellule) ;
4. ouvrir le lien affiché : l'interface Vercel se connecte toute seule au moteur (adresse et clé mémorisées).

Le même carnet fonctionne sur Colab (*Fichier → Importer*, secret dans l'icône 🔑). Le moteur exige une **clé
d'accès** (`CLE_ACCES`) et n'accepte que les sites `localhost` et `*.vercel.app` (`CORS_ORIGINES`, `CORS_REGEX`).
Les vidéos partent en morceaux de 50 Mo (le tunnel refuse les envois de plus de 100 Mo), donc sans limite de taille.
Le voyant d'état en haut à droite ouvre la fenêtre **Connexion au moteur** (bouton *Moteur de ce PC* pour revenir).

À savoir : l'adresse du moteur change à chaque séance, et les fichiers de Kaggle/Colab sont effacés à la fin
(12 h max). Sans sauvegarde automatique (ci-dessous), téléchargez vos vidéos doublées avant.
XTTS v2 (CPML) et NLLB : usage personnel non commercial.

**Sauvegarde automatique sur Hugging Face (recommandé)**

Pour ne rien perdre d'une séance à l'autre, le moteur recopie ses données dans un dépôt **privé** Hugging Face :
les projets terminés (vidéo doublée, `segments.json`, `status.json`, voix, sous-titres…) et la bibliothèque
de voix (aperçus des 51 voix, voix personnelles). La copie se fait après chaque export terminé, quand la
bibliothèque de voix change, et à l'arrêt du moteur ; au début de la séance suivante, le carnet remet tout en
place avant de démarrer le moteur (module `backend/sauvegarde.py`).

1. **Jeton d'écriture** (une seule fois) : huggingface.co → photo de profil → *Settings → Access Tokens* →
   *Create new token* → onglet **Write** → nom : `voxlingo-sauvegarde` → *Create token* → copier le jeton
   (`hf_…`, il ne s'affiche qu'une fois). Le jeton `HF_TOKEN` actuel (lecture seule, pour pyannote) ne suffit
   pas et reste inchangé.
2. **Secret du carnet** : *Kaggle : Add-ons → Secrets* (*Colab : icône 🔑*) → nouveau secret `HF_TOKEN_ECRITURE`
   avec ce jeton, accès autorisé au carnet.
3. **Dépôt privé** : rien à faire, il est créé automatiquement (privé, de type *dataset*) au premier usage :
   `votre-compte/voxlingo-sauvegarde`. Pour le créer vous-même ou choisir un autre nom : huggingface.co →
   *New → Dataset* → nom → **Private** → *Create dataset*, puis renseigner `DEPOT_SAUVEGARDE = "compte/nom"`
   dans la première cellule du carnet. Il est visible sur `https://huggingface.co/datasets/votre-compte/voxlingo-sauvegarde`.

Bon à savoir : un projet supprimé dans l'interface l'est aussi de la sauvegarde ; les paquets Kaggle (`.zip`)
ne sont pas copiés. Pour forcer une sauvegarde complète avant de fermer, exécuter dans une cellule du carnet :
`import subprocess; subprocess.run([sys.executable, "-c", "import sauvegarde; sauvegarde.instance().tout_sauvegarder()"], cwd=CODE / "backend")`. Le jeton n'est jamais affiché dans le carnet ni dans les journaux. En local, la sauvegarde reste
désactivée tant que `HF_TOKEN_ECRITURE` n'est pas défini dans `.env`.

## Autres GPU gratuits : Lightning AI, Modal, toute machine Linux

Le carnet Kaggle n'est pas la seule possibilité. Deux autres moteurs se branchent de la même façon sur
l'interface Vercel (adresse + clé dans la fenêtre **Connexion au moteur**). Leur avantage : un **disque
permanent**, donc projets, voix et modèles ne sont plus effacés à la fin de la séance.

**Modal** (≈ 30 $ de crédits gratuits chaque mois, soit ≈ 50 h de GPU T4) — `deploiement/modal_app.py`
1. modal.com → *Sign up* (compte gratuit), puis sur votre PC : `pip install modal` et `modal token new`
   (connexion dans le navigateur) ;
2. `modal secret create voxlingo CLE_ACCES=<votre mot de passe> HF_TOKEN=<votre jeton Hugging Face>` ;
3. depuis le dossier du projet : `modal deploy deploiement/modal_app.py` (la première fois ≈ 15 min :
   construction de l'image avec Chatterbox et MuseTalk ; refaire après chaque mise à jour du code) ;
4. coller l'adresse affichée (`https://<compte>--voxlingo-moteur-moteur-web.modal.run`) et la clé dans VoxLingo.

Le moteur s'allume à la première requête (≈ 1 à 2 min), reste allumé tant que l'interface est ouverte et
s'éteint 5 min après : éteint, il ne coûte rien. **Gardez l'interface ouverte pendant un doublage** (la
fermer l'interrompt au bout de 5 min). Pour rester dans les crédits gratuits : *Settings → Usage & Billing* →
**Spend limit** à 0 $ (Modal exige une carte pour les GPU, mais ne prélève rien au-delà de cette limite). Projets, voix et modèles sont sur le volume Modal `voxlingo-stockage`.

**Lightning AI** (Studio avec disque permanent ; crédits gratuits à l'inscription) et **toute machine Linux
avec GPU NVIDIA** (RunPod, Vast.ai, serveur personnel) — `deploiement/lancer_moteur.py`
1. lightning.ai → *Sign up* → créer un *Studio* et choisir un GPU (T4 ou L4) ;
2. dans le terminal du Studio : `git clone https://<jeton GitHub>@github.com/Frankynogning237/voxlingo.git`
   (jeton en lecture seule, comme pour Kaggle) puis `cd voxlingo` ;
3. `python deploiement/lancer_moteur.py` : il installe tout (≈ 10 min la première fois, ensuite ≈ 1 min),
   lance le moteur et un tunnel, puis affiche le lien à ouvrir.

Projets, modèles et outils sont gardés à côté du dépôt (`voxlingo-donnees`, `voxlingo-modeles`,
`voxlingo-outils`), la clé d'accès aussi : le lien reste valable d'une séance à l'autre tant que le tunnel
tourne. Réglages possibles : `VOXLINGO_CLE`, `VOXLINGO_STOCKAGE`, `MODELE_TRADUCTION`, `HF_TOKEN`,
`HF_TOKEN_ECRITURE` (voir l'en-tête du script).

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
