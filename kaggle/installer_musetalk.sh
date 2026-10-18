#!/usr/bin/env bash
# Installe MuseTalk 1.5 (synchronisation labiale, licence MIT) dans son propre environnement Python,
# pour le moteur VoxLingo sur Kaggle/Colab. Lancé en arrière-plan par le carnet (≈ 10 min).
#   Environnement : $MUSETALK_RACINE/env (Python 3.10, PyTorch 2.0.1 CUDA 11.8, OpenMMLab)
#   Code et poids : $MUSETALK_RACINE/code ; « .pret » y est créé quand tout est installé.
#   MUSETALK_RACINE vaut /tmp/musetalk par défaut (Kaggle/Colab) ; un disque permanent ailleurs.
#   MUSETALK_SANS_GPU=1 saute la vérification du GPU (construction d'une image sans GPU, Modal).
# Le moteur l'utilise grâce aux réglages MUSETALK_PYTHON et MUSETALK_DIR.
set -euo pipefail

RACINE="${MUSETALK_RACINE:-/tmp/musetalk}"
ENV="$RACINE/env"
CODE="$RACINE/code"
P="$ENV/bin/python"

command -v uv >/dev/null || python3 -m pip install -q uv
uv venv -q --seed --allow-existing -p 3.10 "$ENV"
[ -d "$CODE/.git" ] || git clone -q --depth 1 https://github.com/TMElyralab/MuseTalk.git "$CODE"
cd "$CODE"

echo "PyTorch 2.0.1…"
uv pip install -q -p "$P" torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
echo "Dépendances…"
# TensorFlow, TensorBoard et Gradio ne servent qu'à l'entraînement et à la démo : inutiles ici
grep -viE '^(tensorflow|tensorboard|gradio)' requirements.txt > "$RACINE/requirements.txt"
uv pip install -q -p "$P" -r "$RACINE/requirements.txt" "setuptools<81" "huggingface_hub[cli]" gdown
echo "OpenMMLab (mmcv, mmdet, mmpose)…"
uv pip install -q -p "$P" mmengine "mmcv==2.0.1" \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
uv pip install -q -p "$P" --no-build-isolation chumpy  # dépendance de mmpose, à construire avec pip/setuptools
uv pip install -q -p "$P" "mmdet==3.1.0" "mmpose==1.1.0"

echo "Poids des modèles…"
H="$ENV/bin/huggingface-cli"
"$H" download TMElyralab/MuseTalk musetalkV15/musetalk.json musetalkV15/unet.pth --local-dir models
"$H" download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir models/sd-vae
"$H" download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json --local-dir models/whisper
"$H" download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir models/dwpose
mkdir -p models/face-parse-bisent
[ -s models/face-parse-bisent/79999_iter.pth ] || \
  "$ENV/bin/gdown" -q 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
curl -fsSL https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth

MPLBACKEND=Agg "$P" -c "import os, torch; from mmpose.apis import init_model; assert torch.cuda.is_available() or os.environ.get('MUSETALK_SANS_GPU'), 'pas de GPU'"
touch "$CODE/.pret"
echo "MuseTalk prêt."
