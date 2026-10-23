"""Génère kaggle/infinitetalk_kaggle.ipynb (essai expérimental) et, pour un projet, le paquet à importer sur Kaggle.

Essai de faisabilité : recréer un extrait d'une vidéo doublée d'après la voix doublée (InfiniteTalk, Wan 2.1 14B),
sur les GPU gratuits de Kaggle (T4 x2). InfiniteTalk est conçu pour des cartes récentes (bfloat16,
flash-attention) : le carnet l'adapte à la T4 (float16, attention de PyTorch, modèle compressé, encodeurs sur la
deuxième carte). C'est un essai : il peut échouer ou être très lent, et le carnet le dit clairement.

Usage, depuis la racine du projet :
    python kaggle/generer_essai_infinitetalk.py                      # carnet seulement
    python kaggle/generer_essai_infinitetalk.py 7705009c58d3 59.1 4.7  # carnet + paquet de l'extrait
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
cellules: list[dict] = []


def md(texte: str) -> None:
    cellules.append({"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n").splitlines(keepends=True)})


def code(texte: str) -> None:
    cellules.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": texte.strip("\n").splitlines(keepends=True)})


md("""
# VoxLingo — essai : recréer la vidéo d'après la voix doublée (InfiniteTalk) sur Kaggle

InfiniteTalk refait un extrait de la vidéo d'origine à partir de la voix doublée : **lèvres, expressions et
mouvements de tête suivent la nouvelle voix**, le décor et les personnages restent ceux d'origine.

**Essai expérimental.** Le modèle (14 milliards de paramètres) est conçu pour des cartes récentes ; ce carnet
l'adapte aux T4 gratuites de Kaggle (float16, modèle compressé, deux cartes). Il peut échouer ou être lent :
chaque étape affiche ce qu'il se passe, le rapport final dit où on en est.

**Avant de lancer :**
1. *Session options* → *Accelerator* : **GPU T4 x2** (deux cartes, indispensable).
2. *Internet* : **activé** (environ 32 Go de modèles à télécharger, 15 à 30 minutes).
3. *Add Input* → *Upload* : le fichier **`voxlingo_infinitetalk_<projet>.zip`** préparé par VoxLingo.

Puis *Run All*. À la fin, téléchargez **`comparaison.mp4`** et **`recree.mp4`** (dossier *Output*).

> Usage personnel : ne recréez que des vidéos dont vous avez les droits, avec l'accord des personnes filmées.
""")

code(r"""
# 1. Paquet VoxLingo (extrait d'origine, voix doublée seule, son doublé complet) et matériel disponible
import glob, os, shutil, subprocess, zipfile, time
TRAVAIL = '/kaggle/working/essai'
os.makedirs(TRAVAIL, exist_ok=True)
# Paquet importé (Add Input → Upload) ou récupéré depuis la branche « essai-infinitetalk-donnees » du dépôt
paquets = (glob.glob('/kaggle/input/**/voxlingo_infinitetalk_*.zip', recursive=True)
           + glob.glob('/kaggle/working/donnees/**/voxlingo_infinitetalk_*.zip', recursive=True))
if paquets:
    zipfile.ZipFile(paquets[0]).extractall(TRAVAIL)
else:  # Kaggle décompresse parfois les .zip à l'import
    for f in glob.glob('/kaggle/input/**/*', recursive=True):
        if os.path.basename(f) in ('origine.mp4', 'voix.wav', 'mixage.wav'):
            shutil.copy(f, TRAVAIL)
for nom in ('origine.mp4', 'voix.wav', 'mixage.wav'):
    assert os.path.isfile(os.path.join(TRAVAIL, nom)), f"{nom} introuvable : importez le paquet VoxLingo (Add Input → Upload)."
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
import psutil
print(f"Mémoire vive : {psutil.virtual_memory().total / 1e9:.0f} Go")
# Dossier des modèles : là où il y a le plus de place (il faut ~35 Go)
DOSSIERS = [d for d in ('/kaggle/temp', '/kaggle/tmp', '/tmp', '/kaggle/working') if os.path.isdir(d)]
POIDS = os.path.join(max(DOSSIERS, key=lambda d: shutil.disk_usage(d).free), 'poids_infinitetalk')
os.makedirs(POIDS, exist_ok=True)
print('Modèles dans', POIDS, f"({shutil.disk_usage(POIDS).free / 1e9:.0f} Go libres)")
import torch
assert torch.cuda.device_count() >= 2, "Il faut GPU T4 x2 (Session options → Accelerator)."
""")

code(r"""
# 2. Réglages de l'essai
# « rapide » : modèle compressé int8 accéléré, 8 étapes (conseillé sur T4) ; « qualite » : fp8, 40 étapes (très lent)
MODE = 'rapide'
CONSIGNE = 'A person is talking on the phone, natural facial expressions and head movements'
""")

code(r"""
# 3. Installer InfiniteTalk (sans remplacer le PyTorch de Kaggle) et l'adapter aux cartes T4
%cd /kaggle/working
!rm -rf InfiniteTalk && git clone -q --depth 1 https://github.com/MeiGen-AI/InfiniteTalk
import torch, re
version_torch = torch.__version__.split('+')[0]
# xformers : InfiniteTalk (une personne, une carte) n'en utilise que memory_efficient_attention sans masque ;
# l'attention de PyTorch fait le même calcul. Un petit module de remplacement évite de compiler xformers (plus
# d'une heure quand aucune version toute prête n'existe pour ce PyTorch).
os.makedirs('/kaggle/working/InfiniteTalk/xformers/ops/fmha', exist_ok=True)
def ecrire(chemin, lignes):
    open(f'/kaggle/working/InfiniteTalk/xformers/{chemin}', 'w').write(chr(10).join(lignes) + chr(10))
ecrire('__init__.py', [''])
ecrire('ops/__init__.py', [
    'import torch.nn.functional as F',
    'from . import fmha',
    'def memory_efficient_attention(q, k, v, attn_bias=None, op=None, **_):',
    '    assert attn_bias is None, "masque non pris en charge (plusieurs cartes ou plusieurs personnes)"',
    '    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)',
])
ecrire('ops/fmha/__init__.py', ['from . import attn_bias'])
ecrire('ops/fmha/attn_bias.py', [
    'class BlockDiagonalMask:',
    '    @staticmethod',
    '    def from_seqlens(*a, **k):',
    '        raise NotImplementedError("plusieurs cartes non prises en charge par cet essai")',
])
# xfuser (calcul réparti sur plusieurs cartes) : importé partout, mais une seule carte calcule ici
def module(chemin, lignes):
    os.makedirs(os.path.dirname(f'/kaggle/working/InfiniteTalk/{chemin}'), exist_ok=True)
    open(f'/kaggle/working/InfiniteTalk/{chemin}', 'w').write(chr(10).join(lignes) + chr(10))
module('xfuser/__init__.py', [''])
module('xfuser/core/__init__.py', [''])
module('xfuser/core/distributed/__init__.py', [
    'def get_sequence_parallel_rank(): return 0',
    'def get_sequence_parallel_world_size(): return 1',
    'def get_sp_group(): raise NotImplementedError("une seule carte dans cet essai")',
    'def init_distributed_environment(*a, **k): raise NotImplementedError("une seule carte dans cet essai")',
    'def initialize_model_parallel(*a, **k): raise NotImplementedError("une seule carte dans cet essai")',
])
module('xfuser/core/long_ctx_attention/__init__.py', [
    'class xFuserLongContextAttention:',
    '    def __init__(self, *a, **k): raise NotImplementedError("une seule carte dans cet essai")',
])
# decord (lecture vidéo) n'existe pas pour Python 3.13 : InfiniteTalk n'en lit qu'une image précise de la
# vidéo d'origine, ce qu'OpenCV fait aussi (images RVB, index négatif = depuis la fin)
module('decord/__init__.py', [
    'import cv2',
    'class _Image:',
    '    def __init__(self, tableau): self._t = tableau',
    '    def asnumpy(self): return self._t',
    'def cpu(*a, **k): return None',
    'class VideoReader:',
    '    def __init__(self, chemin, ctx=None, **k):',
    '        self._chemin = chemin',
    '        capture = cv2.VideoCapture(chemin)',
    '        self._num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))',
    '        self._fps = capture.get(cv2.CAP_PROP_FPS)',
    '        capture.release()',
    '    def __len__(self): return self._num_frame',
    '    def get_avg_fps(self): return self._fps',
    '    def __getitem__(self, indice):',
    '        indice = indice % self._num_frame if indice < 0 else indice',
    '        capture = cv2.VideoCapture(self._chemin)',
    '        capture.set(cv2.CAP_PROP_POS_FRAMES, indice)',
    '        lu, image = capture.read()',
    '        capture.release()',
    '        if not lu: raise IndexError(indice)',
    '        return _Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))',
    'class bridge:',
    '    @staticmethod',
    '    def set_bridge(*a, **k): pass',
])
cuda = 'cu' + torch.version.cuda.replace('.', '')[:3]
print('PyTorch', version_torch, cuda)
with open('/tmp/contraintes.txt', 'w') as f:
    f.write(f'torch=={version_torch}' + chr(10))
!grep -v -E "^(xfuser|gradio|dashscope|decord|numpy)" InfiniteTalk/requirements.txt > /tmp/requirements.txt
# transformers 4.51 : la version de l'époque d'InfiniteTalk (la 5.x de Kaggle refuse certains réglages du modèle)
!pip install -q -c /tmp/contraintes.txt -r /tmp/requirements.txt librosa "huggingface_hub[hf_transfer]" "transformers==4.51.3"

# Adaptations T4 (pas de bfloat16 ni de flash-attention ; deux cartes de 16 Go)
def remplacer(fichier, avant, apres, compte=1):
    chemin = f'/kaggle/working/InfiniteTalk/{fichier}'
    texte = open(chemin, encoding='utf-8').read()
    assert texte.count(avant) >= compte, f'{fichier} : « {avant[:60]} » introuvable (le dépôt a changé)'
    open(chemin, 'w', encoding='utf-8').write(texte.replace(avant, apres))
import glob as _g
for f in _g.glob('/kaggle/working/InfiniteTalk/wan/**/*.py', recursive=True):  # float16 partout
    t = open(f, encoding='utf-8').read()
    if 'torch.bfloat16' in t:
        open(f, 'w', encoding='utf-8').write(t.replace('torch.bfloat16', 'torch.float16'))
# Encodeur de texte (compressé fp8) : le seul fourni, quel que soit le mode
remplacer('wan/modules/t5.py', 'f"t5_{quant}.safetensors"', '"t5_fp8.safetensors"', 2)
remplacer('wan/modules/t5.py', 'f"t5_map_{quant}.json"', '"t5_map_fp8.json"', 1)
# Synthèse vocale intégrée (kokoro, pour le mode « texte → voix ») : inutile ici, et ses dépendances manquent
remplacer('generate_infinitetalk.py', 'from kokoro import KPipeline', 'KPipeline = None  # synthèse vocale inutile ici')
# Python 3.13 (Kaggle) : inspect.ArgSpec n'existe plus (importé sans être utilisé)
remplacer('wan/multitalk.py', 'from inspect import ArgSpec', '')
# Modèles compressés : leurs poids sont restitués en bfloat16 ; tout passe en float16 (T4), et l'attention de
# l'encodeur de texte aligne ses précisions (scores en float32, valeurs en float16)
remplacer('wan/modules/t5.py', "            requantize(model, model_state_dict, quantization_map, device='cpu')",
          "            requantize(model, model_state_dict, quantization_map, device='cpu')" + chr(10) +
          "            model = model.to(torch.float16)")
remplacer('wan/multitalk.py', "            requantize(self.model, model_state_dict, quantization_map, device='cpu')",
          "            requantize(self.model, model_state_dict, quantization_map, device='cpu')" + chr(10) +
          "            self.model.to(torch.float16)")
remplacer('wan/modules/t5.py', "x = torch.einsum('bnij,bjnc->binc', attn, v)", "x = torch.einsum('bnij,bjnc->binc', attn.to(v.dtype), v)")
# Encodeur de voix : réglage forcé jamais utilisé ensuite, que les versions récentes de transformers refusent
remplacer('src/audio_analysis/wav2vec2.py', '        self.config.output_attentions = True', '        pass')
# Encodeurs de texte et d'image sur la deuxième carte : la première reste au modèle principal
remplacer('wan/multitalk.py', 'self.text_encoder.model.to(self.device)', 'self.text_encoder.model.to("cuda:1")')
# … dès son chargement et pour toute la séance : la mémoire vive (29 Go) reste au modèle principal (19,5 Go)
remplacer('wan/multitalk.py', '        self.vae_stride = config.vae_stride',
          '        self.text_encoder.model.to("cuda:1")\n        self.vae_stride = config.vae_stride')
remplacer('wan/multitalk.py', 'self.text_encoder.model.cpu()', 'pass  # encodeur gardé sur la deuxième carte')
remplacer('wan/multitalk.py', 'self.clip.model.cpu()', 'pass  # encodeur gardé sur la deuxième carte')
remplacer('wan/multitalk.py', 'context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)',
          'context, context_null = self.text_encoder([input_prompt, n_prompt], torch.device("cuda:1"))\n'
          '            context = [t.to(self.device) for t in context]; context_null = [t.to(self.device) for t in context_null]')
remplacer('wan/multitalk.py', 'self.clip = CLIPModel(\n            dtype=config.clip_dtype,\n            device=self.device,',
          'self.clip = CLIPModel(\n            dtype=config.clip_dtype,\n            device=torch.device("cuda:1"),')
remplacer('wan/multitalk.py', 'self.clip.model.to(self.device)', 'self.clip.model.to("cuda:1")')
remplacer('wan/multitalk.py', 'clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)',
          'clip_context = self.clip.visual(cond_image[:, :, -1:, :, :].to("cuda:1")).to(self.device).to(self.param_dtype)')
print('InfiniteTalk installé et adapté aux T4.')
""")

code(r"""
# 4. Télécharger les modèles compressés (~32 Go, une fois par session)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download, hf_hub_download
debut = time.time()
snapshot_download('Wan-AI/Wan2.1-I2V-14B-480P', local_dir=f'{POIDS}/Wan2.1-I2V-14B-480P',
                  allow_patterns=['config.json', 'Wan2.1_VAE.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                                  'google/*', 'xlm-roberta-large/*'])
snapshot_download('TencentGameMate/chinese-wav2vec2-base', local_dir=f'{POIDS}/chinese-wav2vec2-base')
hf_hub_download('TencentGameMate/chinese-wav2vec2-base', 'model.safetensors', revision='refs/pr/1',
                local_dir=f'{POIDS}/chinese-wav2vec2-base')
MODELE = 'infinitetalk_single_int8_lora' if MODE == 'rapide' else 'infinitetalk_single_fp8'
snapshot_download('MeiGen-AI/InfiniteTalk', local_dir=f'{POIDS}/InfiniteTalk',
                  allow_patterns=[f'quant_models/{MODELE}.*', 'quant_models/t5_fp8.safetensors',
                                  'quant_models/t5_map_fp8.json', 'quant_models/quant.json'])
print(f'Modèles prêts en {(time.time() - debut) / 60:.0f} min ;', f'{shutil.disk_usage(POIDS).free / 1e9:.0f} Go libres')
""")

code(r"""
# 5. Recréer l'extrait d'après la voix doublée
import json
with open(f'{TRAVAIL}/entree.json', 'w') as f:
    json.dump({'prompt': CONSIGNE, 'cond_video': f'{TRAVAIL}/origine.mp4',
               'cond_audio': {'person1': f'{TRAVAIL}/voix.wav'}}, f)
quant = 'int8' if MODE == 'rapide' else 'fp8'
reglages = ('--sample_steps 8 --sample_text_guide_scale 1 --sample_audio_guide_scale 2 --sample_shift 2'
            if MODE == 'rapide' else '--sample_steps 40 --use_teacache')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# jax et TensorFlow, préinstallés sur Kaggle, ne servent pas ici : transformers et diffusers ne les chargent pas
os.environ['USE_FLAX'] = '0'
os.environ['USE_TF'] = '0'
%cd /kaggle/working/InfiniteTalk
debut = time.time()
!python generate_infinitetalk.py --ckpt_dir {POIDS}/Wan2.1-I2V-14B-480P --wav2vec_dir {POIDS}/chinese-wav2vec2-base \
    --infinitetalk_dir {POIDS}/InfiniteTalk/single/infinitetalk.safetensors \
    --quant {quant} --quant_dir {POIDS}/InfiniteTalk/quant_models/{MODELE}.safetensors \
    --input_json {TRAVAIL}/entree.json --size infinitetalk-480 --mode streaming --motion_frame 9 \
    --num_persistent_param_in_dit 0 {reglages} --save_file {TRAVAIL}/recree_brut 2>&1 | tee /kaggle/working/journal.txt | tail -40
CALCUL = time.time() - debut
print(f'Calcul : {CALCUL / 60:.1f} min')
""")

code(r"""
# 6. Résultat : son doublé complet, comparaison côte à côte, rapport
brut = next(iter(glob.glob(f'{TRAVAIL}/recree_brut*.mp4')), None)
duree = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                              f'{TRAVAIL}/origine.mp4'], capture_output=True, text=True).stdout or 0)
rapport = [f'Mode : {MODE}', f'Calcul : {CALCUL / 60:.1f} min pour {duree:.1f} s de vidéo',
           f'Résultat : {"obtenu" if brut else "aucun (voir les messages de la cellule 5)"}']
if brut:
    !ffmpeg -y -v error -i "{brut}" -i "{TRAVAIL}/mixage.wav" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest /kaggle/working/recree.mp4
    !ffmpeg -y -v error -i "{TRAVAIL}/origine.mp4" -i "{brut}" -i "{TRAVAIL}/mixage.wav" -filter_complex "[0:v]scale=-2:480[a];[1:v]scale=-2:480[b];[a][b]hstack=inputs=2[v]" -map "[v]" -map 2:a:0 -c:v libx264 -crf 18 -c:a aac -shortest /kaggle/working/comparaison.mp4
open('/kaggle/working/rapport.txt', 'w').write('\n'.join(rapport))
print('\n'.join(rapport))
# Le dossier de travail (clones, paquet) ne doit pas encombrer l'onglet Output : seuls les résultats y restent
shutil.rmtree('/kaggle/working/voxlingo', ignore_errors=True)
shutil.rmtree('/kaggle/working/donnees', ignore_errors=True)
shutil.rmtree('/kaggle/working/InfiniteTalk', ignore_errors=True)
""")

md("""
## Récupérer le résultat

*Output* → **`comparaison.mp4`** (original à gauche, recréé à droite, son doublé), **`recree.mp4`**, **`rapport.txt`**.

**En cas de problème**
- *Killed* / mémoire vive saturée : relancez la session (les modèles se rechargent) ; si cela se reproduit, l'essai
  dépasse ce que Kaggle offre gratuitement.
- *CUDA out of memory* : raccourcissez l'extrait dans VoxLingo (3 à 4 s).
- Image noire ou bruitée : le calcul en float16 a débordé ; c'est une limite de la T4 pour ce modèle.
""")


def ecrire_carnet() -> Path:
    carnet = {"cells": cellules, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                             "name": "python3"},
                                              "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
    sortie = RACINE / "kaggle" / "infinitetalk_kaggle.ipynb"
    sortie.write_text(json.dumps(carnet, ensure_ascii=False, indent=1), encoding="utf-8")
    return sortie


def preparer_paquet(projet: str, debut: float, duree: float) -> Path:
    """Paquet de l'extrait : vidéo d'origine (sans son), voix doublée seule (16 kHz), son doublé complet."""
    dossier = RACINE / "data" / "projects" / projet
    source = next(dossier.glob("input.*"))
    temp = dossier / "essai_infinitetalk"
    temp.mkdir(exist_ok=True)

    def ffmpeg(*arguments: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{debut}", "-t", f"{duree}", *arguments], check=True)

    ffmpeg("-i", str(source), "-an", "-c:v", "libx264", "-crf", "16", str(temp / "origine.mp4"))
    ffmpeg("-i", str(dossier / "voix_doublee.wav"), "-ac", "1", "-ar", "16000", str(temp / "voix.wav"))
    ffmpeg("-i", str(dossier / "doublage.wav"), str(temp / "mixage.wav"))
    paquet = dossier / f"voxlingo_infinitetalk_{projet}.zip"
    with zipfile.ZipFile(paquet, "w") as z:
        for nom in ("origine.mp4", "voix.wav", "mixage.wav"):
            z.write(temp / nom, nom)
    return paquet


if __name__ == "__main__":
    print("Carnet :", ecrire_carnet())
    if len(sys.argv) >= 2:
        projet = sys.argv[1]
        debut = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        duree = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        print("Paquet à importer sur Kaggle :", preparer_paquet(projet, debut, duree))
