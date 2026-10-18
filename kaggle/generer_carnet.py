"""Génère kaggle/moteur_voxlingo.ipynb (moteur VoxLingo sur GPU gratuit Kaggle ou Colab, derrière un tunnel).

Usage : python kaggle/generer_carnet.py kaggle/moteur_voxlingo.ipynb
Modifiez ce script plutôt que le carnet, puis régénérez le carnet.
"""
import json
import sys
from pathlib import Path

cellules = []


def md(texte):
    cellules.append({"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n").splitlines(keepends=True)})


def code(texte):
    cellules.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": texte.strip("\n").splitlines(keepends=True)})


md("""
# VoxLingo : moteur sur GPU gratuit (Kaggle ou Colab)

Ce carnet fait tourner le **moteur** de VoxLingo (transcription, traduction, voix, export) sur un GPU gratuit,
et le rend joignable par l'interface publiée sur Vercel grâce à un tunnel Cloudflare gratuit.

**À faire une seule fois**
1. **Kaggle** : *Settings → Accelerator → GPU T4 x2* et *Internet → On* (le téléphone doit être vérifié sur votre compte).
   **Colab** : *Exécution → Modifier le type d'exécution → GPU T4*.
2. Ajouter les secrets (*Kaggle : Add-ons → Secrets* ; *Colab : icône 🔑*) : `HF_TOKEN` (votre jeton Hugging Face,
   pour l'identification des locuteurs par pyannote) et `GITHUB_TOKEN` : un jeton GitHub en **lecture seule** sur le dépôt privé `voxlingo`
   (autoriser l'accès au carnet pour chacun).
3. Recommandé : un secret `HF_TOKEN_ECRITURE` (jeton Hugging Face avec droit d'**écriture**) pour la
   **sauvegarde automatique** : projets terminés et voix sont copiés dans votre dépôt privé
   `votre-compte/voxlingo-sauvegarde` et remis en place à la séance suivante (voir le README).
4. Optionnel : un secret `VOXLINGO_CLE` (votre mot de passe d'accès). Sinon une clé est tirée au hasard à chaque séance.
5. Vérifier `ADRESSE_VERCEL` ci-dessous (votre interface en ligne).

**À chaque séance** : *Run All* (≈ 10 min d'installation), puis ouvrir le lien affiché à la fin.
Gardez cet onglet ouvert : la dernière cellule maintient la séance active.

⚠️ Les fichiers disparaissent à la fin de la séance (12 h max sur Kaggle). Avec `HF_TOKEN_ECRITURE`, chaque export
terminé est sauvegardé sur Hugging Face ; sans lui, **téléchargez vos vidéos doublées**.
XTTS v2 (licence CPML) et NLLB : usage **personnel et non commercial** uniquement ; ne clonez une voix qu'avec l'accord de la personne.
""")

code("""
# ── Réglages ──────────────────────────────────────────────────────────────
ADRESSE_VERCEL = "https://voxlingo-gray.vercel.app"  # adresse de votre interface sur Vercel
DEPOT = "Frankynogning237/voxlingo"
BRANCHE = "main"
MODELE_TRADUCTION = "qwen2.5:14b"  # modèle Ollama : le plus fin sur les dialogues (≈ 9 Go, le GPU T4 le fait tourner)
MODELE_VISION = "qwen2.5vl:7b"  # modèle Ollama qui regarde les images pour décrire chaque scène (≈ 6 Go) ; "" = sans
NOM_UTILISATEUR = "Franky"
DEPOT_SAUVEGARDE = ""  # dépôt dataset privé de sauvegarde ; vide = « votre-compte/voxlingo-sauvegarde »
""")

code("""
# ── 1. Secrets, dossiers et récupération du code ──────────────────────────
import os, re, secrets, subprocess, sys, time
from pathlib import Path

SUR_KAGGLE = Path("/kaggle").is_dir()
BASE = Path("/kaggle/working" if SUR_KAGGLE else "/content")
CODE, DONNEES, MODELES = BASE / "voxlingo", BASE / "voxlingo-donnees", Path("/tmp/voxlingo-modeles")


def secret(nom):
    try:
        if SUR_KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(nom)
        from google.colab import userdata
        return userdata.get(nom)
    except Exception:
        return None


def masquer(texte):
    # Ne jamais afficher un jeton GitHub (adresse « x-access-token:… ») ni Hugging Face (« hf_… »)
    texte = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", texte)
    return re.sub(r"hf_[A-Za-z0-9]{6,}", "hf_***", texte)


def run(commande):
    r = subprocess.run(commande, shell=True, text=True, capture_output=True)
    if r.returncode:
        print(masquer(r.stdout[-3000:]), masquer(r.stderr[-3000:]))
        raise RuntimeError(f"Échec : {masquer(commande)[:160]}")
    return r.stdout


jeton = secret("GITHUB_TOKEN")
assert jeton, "Ajoutez le secret GITHUB_TOKEN (voir les instructions en haut)."
# Relance dans la même séance : on garde la clé (le lien déjà ouvert reste valable)
CLE = secret("VOXLINGO_CLE") or globals().get("CLE") or secrets.token_urlsafe(18)

if (CODE / ".git").is_dir():
    # Relance dans la même séance : mise à jour du code (le jeton n'est utilisé que le temps de la commande)
    run(f"git -C {CODE} pull -q https://x-access-token:{jeton}@github.com/{DEPOT}.git {BRANCHE}")
else:
    run(f"git clone -q --depth 1 -b {BRANCHE} https://x-access-token:{jeton}@github.com/{DEPOT}.git {CODE}")
    run(f"git -C {CODE} remote set-url origin https://github.com/{DEPOT}.git")  # le jeton ne reste pas sur le disque
print("Code récupéré :", run(f"git -C {CODE} log -1 --format='%h %s'").strip())
print("GPU :", run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo aucun").strip())
""")

code("""
# ── 2. Installation (≈ 5 à 10 min) ────────────────────────────────────────
run("apt-get -qq update && apt-get -qq install -y ffmpeg zstd lshw > /dev/null")
filtres = subprocess.run("ffmpeg -hide_banner -filters", shell=True, capture_output=True, text=True).stdout
print("FFmpeg Rubber Band :", "oui" if "rubberband" in filtres else "non (voix enfants/seniors un peu moins naturelles)")

# PyTorch déjà installé avec CUDA : on le fige pour que pip ne le remplace pas par une version CPU
run(f"{sys.executable} -m pip freeze | grep -iE '^(torch|torchaudio|torchvision)==' > /tmp/contraintes.txt || true")
# Outils facultatifs (ils demandent un PyTorch récent) : installés à part, le moteur s'en passe s'ils échouent
FACULTATIFS = ("pyannote.audio", "mediapipe", "onnxruntime", "funasr")
lignes = (CODE / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()
(Path("/tmp") / "requirements-base.txt").write_text(
    chr(10).join(l for l in lignes if not l.strip().startswith(FACULTATIFS)), encoding="utf-8")
run(f"{sys.executable} -m pip install -q -c /tmp/contraintes.txt -r /tmp/requirements-base.txt")
print("Dépendances principales installées")
for outil in ("torchcodec",) + FACULTATIFS:
    r = subprocess.run(f"{sys.executable} -m pip install -q -c /tmp/contraintes.txt {outil}",
                       shell=True, text=True, capture_output=True)
    print(f"  {outil} :", "installé" if r.returncode == 0 else "non installé (le moteur s'en passera)")

run("curl -fsSL https://ollama.com/install.sh | sh")
subprocess.run("pkill -f 'ollama serve'", shell=True)  # relance : un seul Ollama
# start_new_session : interrompre une cellule (bouton ■) n'arrête pas ce service
ollama = subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log", "w"), stderr=subprocess.STDOUT,
                          start_new_session=True)
time.sleep(5)
run(f"ollama pull {MODELE_TRADUCTION}")
if MODELE_VISION:  # « regarder » la vidéo : description de chaque scène avant la traduction
    run(f"ollama pull {MODELE_VISION}")
print("Ollama prêt avec", MODELE_TRADUCTION, "et", MODELE_VISION or "sans modèle de vision")
""")

code("""
# ── 3. Démarrage du moteur ────────────────────────────────────────────────
import httpx

# Bibliothèques CUDA fournies par pip (cuBLAS, cuDNN) : nécessaires à faster-whisper sur GPU
libs = []
for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        libs.append(list(__import__(module, fromlist=["_"]).__path__)[0])
    except Exception:
        pass

env = os.environ | {
    "CLE_ACCES": CLE,
    "DATA_DIR": str(DONNEES),
    "MODELS_DIR": str(MODELES),
    "NOM_UTILISATEUR": NOM_UTILISATEUR,
    "OLLAMA_MODEL": MODELE_TRADUCTION,
    "MODELE_VISION": MODELE_VISION or "aucun",
    "COQUI_TOS_AGREED": "1",
    "LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
}
if HF := secret("HF_TOKEN"):
    env["HF_TOKEN"] = HF

# Sauvegarde automatique : on remet en place projets et voix des séances précédentes (une fois par séance)
if HF_ECRITURE := secret("HF_TOKEN_ECRITURE"):
    env["HF_TOKEN_ECRITURE"] = HF_ECRITURE
    if DEPOT_SAUVEGARDE:
        env["HF_DEPOT_SAUVEGARDE"] = DEPOT_SAUVEGARDE
    r = subprocess.run([sys.executable, "-m", "sauvegarde", "restaurer"], cwd=CODE / "backend", text=True,
                       capture_output=True, env=env | {"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    print("Sauvegarde :", masquer((r.stdout.strip() or r.stderr.strip()[-1000:]) or "aucune réponse"))
else:
    print("Sauvegarde automatique désactivée : ajoutez le secret HF_TOKEN_ECRITURE pour ne rien perdre.")

journal = open("/tmp/voxlingo.log", "a")
subprocess.run("pkill -f 'uvicorn main:app'", shell=True)  # relance : l'ancien moteur laisse la place
time.sleep(2)
moteur = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
                          cwd=CODE / "backend", env=env, stdout=journal, stderr=subprocess.STDOUT,
                          start_new_session=True)  # survit à l'interruption d'une cellule
for _ in range(120):
    try:
        sante = httpx.get("http://127.0.0.1:8000/health", headers={"X-Cle-Acces": CLE}, timeout=5).json()
        break
    except Exception:
        if moteur.poll() is not None:
            print(open("/tmp/voxlingo.log").read()[-3000:])
            raise RuntimeError("Le moteur s'est arrêté au démarrage (journal ci-dessus).")
        time.sleep(2)
print("Moteur prêt :", sante["device"], f"{sante['vram_totale_go']} Go de VRAM", "·", sante["modeles"])
""")

code("""
# ── 4. Tunnel public gratuit (Cloudflare) et lien de connexion ────────────
from IPython.display import HTML, display
from urllib.parse import quote

ancien = globals().get("tunnel")
if ancien is not None and ancien.poll() is None and globals().get("ADRESSE_MOTEUR"):
    # Relance dans la même séance : le tunnel tourne toujours, l'adresse (et le lien) ne changent pas
    print("Tunnel conservé :", ADRESSE_MOTEUR)
else:
    run("wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared")
    subprocess.run("pkill -f cloudflared", shell=True)  # un seul tunnel
    open("/tmp/tunnel.log", "w").close()
    tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                              stdout=open("/tmp/tunnel.log", "w"), stderr=subprocess.STDOUT,
                              start_new_session=True)  # survit à l'interruption d'une cellule
    ADRESSE_MOTEUR = None
for _ in range(60 if not ADRESSE_MOTEUR else 0):
    trouve = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open("/tmp/tunnel.log").read())
    if trouve:
        ADRESSE_MOTEUR = trouve.group(0)
        break
    time.sleep(2)
assert ADRESSE_MOTEUR, "Le tunnel n'a pas démarré : relancez cette cellule."

lien = f"{ADRESSE_VERCEL.rstrip('/')}/?moteur={quote(ADRESSE_MOTEUR, safe='')}&cle={quote(CLE, safe='')}#/"
display(HTML(f'''<div style="font-size:16px;line-height:1.8">
<b>✅ Moteur en ligne.</b><br>
👉 <a href="{lien}" target="_blank"><b>Ouvrir VoxLingo connecté à ce moteur</b></a><br>
Adresse du moteur : <code>{ADRESSE_MOTEUR}</code><br>
Clé d'accès : <code>{CLE}</code><br>
<small>Ne partagez ce lien qu'avec vous-même : il contient la clé.</small></div>'''))
""")

code("""
# ── 5. Garder la séance active et suivre le journal ───────────────────────
# Laissez cette cellule tourner pendant vos doublages ; arrêtez-la (■) pour terminer.
try:
    while True:
        time.sleep(60)
        etat = "en marche" if moteur.poll() is None else "ARRÊTÉ"
        derniere = (open("/tmp/voxlingo.log").read().strip().splitlines() or [""])[-1]
        print(time.strftime("%H:%M"), "· moteur", etat, "·", derniere[-150:], flush=True)
except KeyboardInterrupt:
    print("Suivi arrêté (le moteur reste lancé tant que la séance est ouverte).")
""")

code("""
# ── 6. Facultatif : sauvegarder tout de suite avant de fermer ─────────────
# (arrêtez d'abord la cellule 5 avec ■, puis lancez celle-ci)
# Chaque export terminé est déjà sauvegardé automatiquement ; cette cellule recopie en plus tous les
# projets terminés et la bibliothèque de voix (utile après avoir modifié des voix juste avant de partir).
if env.get("HF_TOKEN_ECRITURE"):
    r = subprocess.run([sys.executable, "-c", "import sauvegarde; sauvegarde.instance().tout_sauvegarder()"],
                       cwd=CODE / "backend", text=True, capture_output=True,
                       env=env | {"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    print(masquer(r.stderr.strip()[-1500:]) or "Sauvegarde terminée.")
else:
    print("Pas de secret HF_TOKEN_ECRITURE : rien n'est sauvegardé.")
""")

carnet = {
    "cells": cellules,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path(sys.argv[1]).write_text(json.dumps(carnet, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("carnet écrit :", sys.argv[1])
