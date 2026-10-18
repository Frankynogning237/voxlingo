"""Essai de la synchronisation labiale par le moteur (MuseTalk) sur Kaggle/Colab, depuis le carnet :

    exec(open(CODE / "kaggle" / "essai_synchro.py", encoding="utf-8").read())

Double un extrait de 12 s de la vidéo de démonstration de MuseTalk (un visage qui parle, de face)
avec l'option « Synchronisation labiale », puis affiche le résultat. Tout se passe sur le serveur ;
la clé d'accès reste dans le carnet (variable CLE).
"""
import subprocess as _sp
import time as _time
from pathlib import Path as _Path

import httpx as _httpx

_api = _httpx.Client(base_url="http://127.0.0.1:8000", headers={"X-Cle-Acces": CLE}, timeout=120)
_pret = _Path("/tmp/musetalk/code/.pret")


def _etape(texte):
    print(_time.strftime("%H:%M:%S"), texte, flush=True)


# ── 1. MuseTalk installé ? ────────────────────────────────────────────────
if not _pret.exists():
    if _sp.run("pgrep -f 'installer[_]musetalk'", shell=True, capture_output=True).returncode != 0:
        _sp.Popen(["bash", str(CODE / "kaggle" / "installer_musetalk.sh")], stdout=open("/tmp/musetalk.log", "w"),
                  stderr=_sp.STDOUT, start_new_session=True)
        _etape("MuseTalk : installation lancée")
    _fin = _time.time() + 35 * 60
    while not _pret.exists() and _time.time() < _fin:
        _time.sleep(60)
        _derniere = (open("/tmp/musetalk.log", errors="replace").read().strip().splitlines() or [""])[-1]
        _etape(f"  installation… {masquer(_derniere)[:150]}")
        if _sp.run("pgrep -f 'installer[_]musetalk'", shell=True, capture_output=True).returncode != 0:
            break
if not _pret.exists():
    print(masquer(open("/tmp/musetalk.log", errors="replace").read())[-3000:])
    raise RuntimeError("MuseTalk n'est pas installé (journal ci-dessus).")
_etape("MuseTalk prêt ; le moteur le voit : " + str(_api.get("/health").json()["modeles"]["synchro_labiale_locale"]))

# ── 2. Vidéo d'essai : un visage qui parle ────────────────────────────────
_video = _Path("/tmp/essai_synchro.mp4")
run(f"ffmpeg -y -v error -i /tmp/musetalk/code/data/video/yongen.mp4 -t 12 -c:v libx264 -c:a aac {_video}")

# ── 3. Doublage avec synchronisation labiale ──────────────────────────────
with open(_video, "rb") as _f:
    _id = _api.post("/jobs", files={"video": ("essai_synchro.mp4", _f, "video/mp4")},
                    data={"langue_cible": "en", "synchro_labiale": "true"}).json()["id"]
_etape(f"Projet {_id} créé")


def _attendre(etats, minutes):
    derniere, fin = None, _time.time() + minutes * 60
    while _time.time() < fin:
        s = _api.get(f"/jobs/{_id}").json()
        if (s.get("etape"), s.get("progression_etape")) != derniere:
            derniere = (s.get("etape"), s.get("progression_etape"))
            _etape(f"  {s.get('etape')} {s.get('progression_etape')} % ({s['etat']})")
        if s["etat"] in etats:
            return s
        _time.sleep(10)
    return s


_s = _attendre({"a_valider", "erreur", "annule"}, 30)
if _s["etat"] == "a_valider":
    _segments = _api.get(f"/jobs/{_id}/segments").json()["segments"]
    for _x in _segments[:4]:
        print(f"    {_x.get('texte', '')}  →  {_x.get('traduction', '')}")
    _api.post(f"/jobs/{_id}/valider",
              json={"segments": [{"id": x["id"], "traduction": x["traduction"]} for x in _segments]}).raise_for_status()
    _s = _attendre({"termine", "erreur", "annule"}, 45)

# ── 4. Rapport ────────────────────────────────────────────────────────────
_synchro = next(e for e in _s["etapes"] if e["cle"] == "synchro_labiale")
print()
_etape(f"RÉSULTAT : {_s['etat']} · synchro : {_synchro['etat']} ({_synchro.get('duree_s')} s) · "
       f"par le moteur : {_s.get('synchro_moteur')} · erreur : {_s.get('synchro_erreur')}")
for _fichier in sorted((DONNEES / "projects" / _id).glob("doublage_synchro.*")):
    _duree = run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {_fichier}").strip()
    _etape(f"Vidéo synchronisée : {_fichier.name}, {_fichier.stat().st_size // 1024} Ko, {_duree} s")
print("── Journal (synchronisation) ──")
for _l in open("/tmp/voxlingo.log", encoding="utf-8", errors="replace").read().splitlines()[-400:]:
    if "lipsync" in _l or "MuseTalk" in _l or "Synchronisation" in _l or ("ERROR" in _l and _id in _l):
        print(masquer(_l)[:230])
