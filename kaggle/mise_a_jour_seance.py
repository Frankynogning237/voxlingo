"""Met à jour une séance Kaggle/Colab déjà lancée, sans réimporter le carnet. Depuis le carnet :

    run(f"git -C {CODE} pull -q https://x-access-token:{jeton}@github.com/{DEPOT}.git {BRANCHE}")
    exec(open(CODE / "kaggle" / "mise_a_jour_seance.py", encoding="utf-8").read())

1. installe en arrière-plan ce qui manque : Chatterbox (voix) et MuseTalk (synchronisation labiale) ;
2. redémarre le moteur avec le nouveau code et leurs réglages (seulement si aucun doublage n'est en
   cours) ; le tunnel et le lien de connexion ne changent pas.
"""
import subprocess as _sp
import time as _time
from pathlib import Path as _Path

import httpx as _httpx

_entete = {"X-Cle-Acces": CLE}


def _tourne(motif):
    return _sp.run(f"pgrep -f '{motif}'", shell=True, capture_output=True).returncode == 0


# ── 1. Installations en arrière-plan ──────────────────────────────────────
if not _Path("/tmp/chatterbox/.pret").exists() and not _tourne("chatterbox[-]tts"):
    _sp.Popen(f"{sys.executable} -m pip install -q uv && uv venv -q --allow-existing -p 3.11 /tmp/chatterbox"
              " && uv pip install -q -p /tmp/chatterbox/bin/python chatterbox-tts 'setuptools<81'"
              " && touch /tmp/chatterbox/.pret",
              shell=True, stdout=open("/tmp/chatterbox.log", "w"), stderr=_sp.STDOUT, start_new_session=True)
    print("Chatterbox : installation lancée (≈ 5 min)")
if not _Path("/tmp/musetalk/code/.pret").exists() and not _tourne("installer[_]musetalk"):
    _sp.Popen(["bash", str(CODE / "kaggle" / "installer_musetalk.sh")], stdout=open("/tmp/musetalk.log", "w"),
              stderr=_sp.STDOUT, start_new_session=True)
    print("MuseTalk : installation lancée (≈ 5 à 10 min)")

# ── 2. Redémarrage du moteur avec les nouveaux réglages ───────────────────
_occupes = [j["id"] for j in _httpx.get("http://127.0.0.1:8000/jobs", headers=_entete, timeout=10).json()
            if j["etat"] in ("en_cours", "en_attente")]
if _occupes:
    print("Doublage en cours : le moteur n'est pas redémarré (relancez cette cellule plus tard).", _occupes)
else:
    env.update(CHATTERBOX_PYTHON="/tmp/chatterbox/bin/python", MUSETALK_PYTHON="/tmp/musetalk/env/bin/python",
               MUSETALK_DIR="/tmp/musetalk/code", MPLBACKEND="Agg")
    _sp.run("pkill -f 'uvicorn main:app'", shell=True)
    _time.sleep(3)
    moteur = _sp.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
                       cwd=CODE / "backend", env=env, stdout=journal, stderr=_sp.STDOUT, start_new_session=True)
    for _ in range(60):
        try:
            _sante = _httpx.get("http://127.0.0.1:8000/health", headers=_entete, timeout=5).json()
            break
        except Exception:
            _time.sleep(2)
    print("Moteur redémarré :", _sante["modeles"])
