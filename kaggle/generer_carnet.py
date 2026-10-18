"""Génère kaggle/moteur_voxlingo.ipynb : le lanceur du moteur VoxLingo sur GPU gratuit (Kaggle ou Colab).

Usage : python kaggle/generer_carnet.py kaggle/moteur_voxlingo.ipynb

Le carnet ne fait que récupérer la dernière version du code puis exécuter kaggle/demarrer.py (qui
s'appuie sur deploiement/lancer_moteur.py). Toute amélioration poussée sur GitHub est donc prise en compte
au prochain « Run All », sans réimporter le carnet. Le code des cellules reste à plat (sans blocs
indentés) pour pouvoir aussi être retapé tel quel dans l'éditeur de Kaggle.
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

Ce carnet fait tourner le **moteur** de VoxLingo (transcription, traduction, voix, synchronisation labiale)
sur un GPU gratuit, joignable par l'interface publiée sur Vercel grâce à un tunnel Cloudflare gratuit.
Il récupère à chaque lancement **la dernière version du code** : inutile de le réimporter après une mise à jour.

**À faire une seule fois**
1. **Kaggle** : *Settings → Accelerator → GPU T4 x2* et *Internet → On* (téléphone vérifié sur le compte).
   **Colab** : *Exécution → Modifier le type d'exécution → GPU T4*.
2. Secrets (*Kaggle : Add-ons → Secrets* ; *Colab : icône 🔑*), en autorisant l'accès au carnet :
   - `GITHUB_TOKEN` (obligatoire) : jeton GitHub en **lecture seule** sur le dépôt privé `voxlingo` ;
   - `HF_TOKEN` (conseillé) : jeton Hugging Face, pour reconnaître les différents locuteurs ;
   - `HF_TOKEN_ECRITURE` (conseillé) : jeton Hugging Face avec droit d'**écriture**, pour la **sauvegarde
     automatique** des projets et des voix (rien n'est perdu à la fin de la séance, voir le README) ;
   - `VOXLINGO_CLE` (facultatif) : votre mot de passe d'accès ; sinon une clé est tirée au hasard.

**À chaque séance** : *Run All* (≈ 10 min la première fois), puis ouvrir le lien affiché.
Gardez cet onglet ouvert : la dernière cellule maintient la séance active (12 h max sur Kaggle).

XTTS v2 (licence CPML) et NLLB : usage **personnel et non commercial** ; ne clonez une voix qu'avec l'accord
de la personne.
""")

code("""
# ── Réglages ──────────────────────────────────────────────────────────────
ADRESSE_VERCEL = "https://voxlingo-gray.vercel.app"  # adresse de votre interface sur Vercel
DEPOT = "Frankynogning237/voxlingo"
BRANCHE = "main"
MODELE_TRADUCTION = "qwen2.5:14b"  # modèle Ollama ; "qwen3:14b" se choisit aussi dans les Paramètres
MODELE_VISION = "qwen2.5vl:7b"  # description des scènes ; "aucun" pour désactiver
""")

code("""
# ── Dernière version du code, puis démarrage du moteur (installation, tunnel, lien) ──
import subprocess
from pathlib import Path
SUR_KAGGLE = Path("/kaggle").is_dir()
CODE = Path("/kaggle/working/voxlingo" if SUR_KAGGLE else "/content/voxlingo")
jeton = __import__("kaggle_secrets").UserSecretsClient().get_secret("GITHUB_TOKEN") if SUR_KAGGLE else __import__("google.colab.userdata", fromlist=["get"]).get("GITHUB_TOKEN")
depot = f"https://x-access-token:{jeton}@github.com/{DEPOT}.git"
commande = ["git", "-C", str(CODE), "pull", "-q", depot, BRANCHE] if (CODE / ".git").is_dir() else ["git", "clone", "-q", "--depth", "1", "-b", BRANCHE, depot, str(CODE)]
recuperation = subprocess.run(commande, capture_output=True, text=True)
del jeton, depot, commande  # le jeton ne reste ni en mémoire ni sur le disque
assert recuperation.returncode == 0, "Récupération du code impossible : vérifiez le secret GITHUB_TOKEN (lecture du dépôt)."
subprocess.run(["git", "-C", str(CODE), "remote", "set-url", "origin", f"https://github.com/{DEPOT}.git"])
print("Code :", subprocess.run(["git", "-C", str(CODE), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip())
exec(open(CODE / "kaggle" / "demarrer.py", encoding="utf-8").read())
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
