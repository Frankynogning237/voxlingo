"""Essai complet du moteur sur Kaggle/Colab, à lancer depuis le carnet une fois le moteur en ligne :

    exec(open(CODE / "kaggle" / "essai_moteur.py", encoding="utf-8").read())

Tout se passe sur le serveur (rien ne transite par la connexion de l'utilisateur), et la clé
d'accès reste dans le carnet (variable CLE), elle n'est jamais affichée.

1. installe Chatterbox s'il ne l'est pas encore ;
2. génère les aperçus de la bibliothèque de voix (XTTS) : ce sont de vraies voix françaises ;
3. fabrique une vidéo d'essai de ~20 s : deux voix (femme, homme) sur une musique de fond ;
4. la double en anglais avec Qwen3 14B, puis affiche ce que le moteur a utilisé
   (RoFormer ou Demucs, Chatterbox ou XTTS, modèle de traduction) et les traductions.
"""
import json as _json
import subprocess as _sp
import time as _time
from pathlib import Path as _Path

import httpx as _httpx

# Carnet lanceur (kaggle/demarrer.py) : les outils viennent de deploiement/lancer_moteur.py
if "run" not in globals() or "DONNEES" not in globals():
    import os as _os
    import sys as _sys

    _sys.path.insert(0, str(CODE / "deploiement"))
    import lancer_moteur as _lm

    run, masquer, sys = _lm.run, _lm.masquer, _sys
    DONNEES = _Path(_os.environ.get("VOXLINGO_DONNEES", "/kaggle/working/voxlingo-donnees"))

_api = _httpx.Client(base_url="http://127.0.0.1:8000", headers={"X-Cle-Acces": CLE}, timeout=120)
_essai = _Path("/tmp/essai")
_essai.mkdir(exist_ok=True)


def _etape(texte):
    print(_time.strftime("%H:%M:%S"), texte, flush=True)


def _attendre(condition, minutes, message):
    fin = _time.time() + minutes * 60
    while _time.time() < fin:
        resultat = condition()
        if resultat:
            return resultat
        _time.sleep(10)
    _etape(f"⚠️ {message} (plus de {minutes} min)")
    return None


# ── 1. Chatterbox (environnement séparé, en arrière-plan) ──────────────────
_pret = _Path("/tmp/chatterbox/.pret")
if _pret.exists():
    _etape("Chatterbox déjà installé")
elif _sp.run("pgrep -f 'chatterbox[-]tts'", shell=True, capture_output=True).returncode != 0:
    _sp.Popen(f"{sys.executable} -m pip install -q uv && uv venv -q --allow-existing -p 3.11 /tmp/chatterbox"
              " && uv pip install -q -p /tmp/chatterbox/bin/python chatterbox-tts 'setuptools<81' && touch /tmp/chatterbox/.pret",
              shell=True, stdout=open("/tmp/chatterbox.log", "w"), stderr=_sp.STDOUT, start_new_session=True)
    _etape("Chatterbox : installation lancée en arrière-plan")

# ── 2. Aperçus de la bibliothèque de voix ─────────────────────────────────
_etape("Aperçus des voix (XTTS)…")
_api.post("/voix/apercus")
_attendre(lambda: not _api.get("/voix").json()["apercus"]["en_cours"], 25, "Aperçus trop longs")
_banque = _api.get("/voix").json()
if _banque["apercus"].get("erreur"):
    _etape("Aperçus : " + masquer(str(_banque["apercus"]["erreur"]))[:300])
_voix = [v for v in _banque["voix"] if v.get("apercu") and v.get("age") == "adulte"]
_femme = next(v for v in _voix if v["sexe"] == "femme")
_homme = next(v for v in _voix if v["sexe"] == "homme")
_etape(f"{sum(1 for v in _banque['voix'] if v.get('apercu'))} aperçus prêts ; essai avec {_femme['nom']} et {_homme['nom']}")

# ── 3. Vidéo d'essai : deux voix sur une musique ──────────────────────────
_apercus = DONNEES / "voix" / "apercus"
_video = _essai / "essai.mp4"
run(
    f"ffmpeg -y -v error -i {_apercus / (_femme['id'] + '.wav')} -i {_apercus / (_homme['id'] + '.wav')} "
    "-f lavfi -i anullsrc=r=24000:cl=mono -filter_complex "
    "\"[2]asplit[n1][n2];[n1]atrim=0:0.8[s1];[n2]atrim=0:0.8[s2];[0][s1][1][s2]concat=n=4:v=0:a=1,aresample=44100,"
    "pan=stereo|c0=c0|c1=c0[voix];"
    "aevalsrc='0.06*sin(2*PI*220*t)+0.05*sin(2*PI*277*t)+0.04*sin(2*PI*330*t)':s=44100:c=stereo[musique];"
    "[voix][musique]amix=inputs=2:duration=first:normalize=0[a]\" "
    "-f lavfi -i color=c=0x1e1b4b:s=640x360:r=25 -map 3:v -map [a] -shortest "
    f"-c:v libx264 -pix_fmt yuv420p -c:a aac {_video}"
)
_etape(f"Vidéo d'essai : {_video.stat().st_size // 1024} Ko")

# ── 4. Réglages : Qwen3 14B, moteur de voix automatique ───────────────────
_reglages = _api.get("/parametres").json()["valeurs"]
_reglages.update(ollama_model="qwen3:14b", moteur_voix="auto")
_api.put("/parametres", json=_reglages).raise_for_status()
_etape("Réglages : traduction qwen3:14b, voix automatique (Chatterbox s'il est prêt)")

# ── 5. Doublage français → anglais ────────────────────────────────────────
with open(_video, "rb") as _f:
    _job = _api.post("/jobs", files={"video": ("essai.mp4", _f, "video/mp4")},
                     data={"langue_cible": "en", "conserver_fond": "true"}).json()
_id = _job["id"]
_etape(f"Projet {_id} créé ; transcription, séparation, analyse, traduction…")


def _suivre(etats_finaux, minutes):
    derniere = None

    def verifier():
        nonlocal derniere
        s = _api.get(f"/jobs/{_id}").json()
        if s.get("etape") != derniere:
            derniere = s.get("etape")
            _etape(f"  étape : {derniere} ({s['etat']})")
        return s if s["etat"] in etats_finaux else None

    return _attendre(verifier, minutes, "Traitement trop long")


_statut = _suivre({"a_valider", "erreur", "annule", "termine"}, 40)
if _statut and _statut["etat"] == "a_valider":
    _segments = _api.get(f"/jobs/{_id}/segments").json()["segments"]
    _etape(f"Traduction ({_statut.get('moteur_traduction')}) :")
    for _s in _segments[:6]:
        print(f"    FR : {_s.get('texte', '')}\n    EN : {_s.get('traduction', '')}")
    if not _pret.exists():
        _etape("Attente de Chatterbox avant de générer les voix…")
        _attendre(_pret.exists, 20, "Chatterbox pas encore prêt : XTTS sera utilisé")
    _api.post(f"/jobs/{_id}/valider", json={"segments": [{"id": s["id"], "traduction": s["traduction"]}
                                                         for s in _segments]}).raise_for_status()
    _etape("Voix et export…")
    _statut = _suivre({"termine", "erreur", "annule"}, 30)

# ── 6. Rapport ────────────────────────────────────────────────────────────
print()
_etape(f"RÉSULTAT : {_statut and _statut['etat']} {masquer(str(_statut.get('erreur') or '')) if _statut else ''}")
_sortie = DONNEES / "projects" / _id / "doublage.mp4"
if _sortie.is_file():
    _duree = run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {_sortie}").strip()
    _etape(f"Vidéo doublée : {_sortie.stat().st_size // 1024} Ko, {_duree} s")

# Fidélité : niveau de chaque phrase d'origine et niveau obtenu par la voix doublée
_sys_path = str(CODE / "backend")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from pipeline.niveau import niveau_fichier_db as _niveau  # noqa: E402

print("── Volume (dB) : origine → doublage ──")
_ecarts = []
for _s in _json.loads((DONNEES / "projects" / _id / "segments.json").read_text(encoding="utf-8"))["segments"]:
    _cale = DONNEES / "projects" / _id / "voix" / f"cale_{_s['id']:05d}.wav"
    _obtenu = _niveau(_cale) if _cale.is_file() else None
    if _s.get("niveau_origine_db") is not None and _obtenu is not None:
        _ecarts.append(abs(_obtenu - _s["niveau_origine_db"]))
    print(f"    phrase {_s['id']} ({_s.get('locuteur')}) : {_s.get('niveau_origine_db')} → {_obtenu} "
          f"(gain {_s.get('gain_db')} dB)")
if _ecarts:
    _etape(f"Écart de volume moyen : {sum(_ecarts) / len(_ecarts):.2f} dB (max {max(_ecarts):.2f} dB)")

_mots = ("RoFormer", "Demucs", "Chatterbox", "XTTS", "ollama", "qwen3", "Modèle", "Voix :", "ERROR", "Traceback",
         "Error", "échoué", "indisponible", "Timbre", "Relecture")
_journal = [l for l in open("/tmp/voxlingo.log", encoding="utf-8", errors="replace").read().splitlines()
            if _id in l or any(m in l for m in _mots)]
print("── Journal du moteur (extraits) ──")
for _l in _journal[-45:]:
    print(masquer(_l)[:220])
