"""Démarre le moteur VoxLingo depuis le carnet Kaggle ou Colab (exécuté par le carnet, après la
récupération du code). Toute la logique vit ici et dans deploiement/lancer_moteur.py : une amélioration
poussée sur GitHub est prise en compte au prochain « Run All », sans réimporter le carnet.

Variables lues dans le carnet si elles existent : CODE, ADRESSE_VERCEL, MODELE_TRADUCTION, MODELE_VISION.
Secrets lus (Kaggle : Add-ons → Secrets ; Colab : icône 🔑), tous facultatifs :
HF_TOKEN, HF_TOKEN_ECRITURE, HF_DEPOT_SAUVEGARDE, VOXLINGO_CLE.
"""
import importlib as _importlib
import os as _os
import sys as _sys
from pathlib import Path as _Path

from IPython.display import HTML as _HTML, display as _display

_KAGGLE = _Path("/kaggle").is_dir()


def _secret(nom):
    try:
        if _KAGGLE:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(nom)
        from google.colab import userdata
        return userdata.get(nom)
    except Exception:
        return None


for _nom in ("HF_TOKEN", "HF_TOKEN_ECRITURE", "HF_DEPOT_SAUVEGARDE", "VOXLINGO_CLE"):
    if _valeur := _secret(_nom):
        _os.environ[_nom] = _valeur
if not _os.environ.get("VOXLINGO_CLE") and globals().get("CLE"):
    _os.environ["VOXLINGO_CLE"] = CLE  # relance dans la même séance : le lien déjà ouvert reste valable
for _nom in ("ADRESSE_VERCEL", "MODELE_TRADUCTION", "MODELE_VISION"):
    if globals().get(_nom):
        _os.environ[_nom] = globals()[_nom]
# Projets dans le dossier de travail ; modèles et outils dans /tmp (le dossier de travail est limité à 20 Go)
_os.environ.setdefault("VOXLINGO_DONNEES", "/kaggle/working/voxlingo-donnees" if _KAGGLE else "/content/voxlingo-donnees")
_os.environ.setdefault("VOXLINGO_MODELES", "/tmp/voxlingo-modeles")
_os.environ.setdefault("VOXLINGO_OUTILS", "/tmp")
_os.environ["DATA_DIR"] = _os.environ["VOXLINGO_DONNEES"]  # pour la sauvegarde manuelle depuis le carnet

_code = _Path(globals().get("CODE") or ("/kaggle/working/voxlingo" if _KAGGLE else "/content/voxlingo"))
_sys.path.insert(0, str(_code / "deploiement"))
import lancer_moteur  # noqa: E402

_importlib.reload(lancer_moteur)  # relance après une mise à jour du code
moteur, ADRESSE_MOTEUR, CLE = lancer_moteur.lancer()
_lien = lancer_moteur.lien_de_connexion(ADRESSE_MOTEUR, CLE)
_display(_HTML(f'''<div style="font-size:16px;line-height:1.8">
<b>✅ Moteur en ligne.</b><br>
👉 <a href="{_lien}" target="_blank"><b>Ouvrir VoxLingo connecté à ce moteur</b></a><br>
Adresse du moteur : <code>{ADRESSE_MOTEUR}</code><br>
Clé d'accès : <code>{CLE}</code><br>
<small>Ne partagez ce lien qu'avec vous-même : il contient la clé. Gardez cet onglet ouvert :
cette cellule maintient la séance active.</small></div>'''))
lancer_moteur.suivre(moteur)
