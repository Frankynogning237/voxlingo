"""Tests du lanceur deploiement/lancer_moteur.py (fonctions pures)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_chemin = Path(__file__).resolve().parents[2] / "deploiement" / "lancer_moteur.py"
_spec = importlib.util.spec_from_file_location("lancer_moteur", _chemin)
lancer_moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lancer_moteur)


def test_motif_pgrep_ne_se_reconnait_pas_lui_meme():
    assert lancer_moteur.motif_pgrep("ollama serve") == "[o]llama serve"
    assert lancer_moteur.motif_pgrep("chatterbox[-]tts") == "chatterbox[-]tts"  # déjà protégé


@pytest.mark.skipif(sys.platform == "win32", reason="pgrep n'existe que sous Linux/macOS")
def test_tourne_faux_pour_un_processus_absent():
    assert not lancer_moteur.tourne("processus-voxlingo-inexistant serve")
    with subprocess.Popen(["sleep", "5"]):
        assert lancer_moteur.tourne("sleep 5")


def test_masquer_les_jetons():
    texte = "https://x-access-token:ghp_abc123@github.com et hf_AbCdEf123456"
    assert lancer_moteur.masquer(texte) == "https://x-access-token:***@github.com et hf_***"
    journal = '"GET /jobs/ab12/video?cle=Xy_9-z&t=1 HTTP/1.1" 206 · /thumbnail?cle=Q2w HTTP'
    assert lancer_moteur.masquer(journal) == '"GET /jobs/ab12/video?cle=***&t=1 HTTP/1.1" 206 · /thumbnail?cle=*** HTTP'


def test_lien_de_connexion_encode_adresse_et_cle():
    lien = lancer_moteur.lien_de_connexion("https://abc.trycloudflare.com", "cl/é+1")
    assert lien.startswith(lancer_moteur.ADRESSE_VERCEL.rstrip("/") + "/?moteur=https%3A%2F%2Fabc.trycloudflare.com")
    assert "cle=cl%2F%C3%A9%2B1" in lien and lien.endswith("#/")
