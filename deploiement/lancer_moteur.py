"""Lance le moteur VoxLingo sur n'importe quelle machine Linux avec GPU NVIDIA (Lightning AI Studio,
RunPod, Vast.ai, serveur personnel…), joignable par l'interface Vercel grâce à un tunnel Cloudflare.

Même logique que le carnet Kaggle (kaggle/generer_carnet.py), mais en script, avec un disque permanent :
projets, modèles téléchargés et outils installés restent d'une séance à l'autre.

Usage, depuis le dossier du dépôt cloné :
    python deploiement/lancer_moteur.py
Réglages facultatifs (variables d'environnement) :
    VOXLINGO_CLE          clé d'accès (sinon tirée au hasard une fois, puis gardée dans le dossier des données)
    VOXLINGO_STOCKAGE     dossier permanent (défaut : le dossier qui contient le dépôt)
    VOXLINGO_DONNEES, VOXLINGO_MODELES, VOXLINGO_OUTILS   un dossier pour chacun (Kaggle : modèles et outils
                          dans /tmp, plus grand que le dossier de travail limité à 20 Go)
    MODELE_TRADUCTION     modèle Ollama (défaut : qwen2.5:14b) ; MODELE_VISION (défaut : qwen2.5vl:7b)
    ADRESSE_VERCEL        adresse de l'interface (défaut : https://voxlingo-gray.vercel.app)
    HF_TOKEN, HF_TOKEN_ECRITURE, HF_DEPOT_SAUVEGARDE   comme dans le carnet Kaggle
Ctrl+C arrête le suivi ; le moteur et le tunnel continuent jusqu'à `pkill -f "uvicorn main:app"`.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

CODE = Path(__file__).resolve().parent.parent
STOCKAGE = Path(os.environ.get("VOXLINGO_STOCKAGE") or CODE.parent)
DONNEES = Path(os.environ.get("VOXLINGO_DONNEES") or STOCKAGE / "voxlingo-donnees")
MODELES = Path(os.environ.get("VOXLINGO_MODELES") or STOCKAGE / "voxlingo-modeles")
OUTILS = Path(os.environ.get("VOXLINGO_OUTILS") or STOCKAGE / "voxlingo-outils")
ADRESSE_VERCEL = os.environ.get("ADRESSE_VERCEL", "https://voxlingo-gray.vercel.app")
MODELE_TRADUCTION = os.environ.get("MODELE_TRADUCTION", "qwen2.5:14b")
MODELE_VISION = os.environ.get("MODELE_VISION", "qwen2.5vl:7b")
# Outils facultatifs : installés à part, le moteur s'en passe s'ils échouent
FACULTATIFS = ("pyannote.audio", "mediapipe", "onnxruntime", "audio-separator", "funasr")


def masquer(texte: str) -> str:
    """Ne jamais afficher un jeton (GitHub, Hugging Face)."""
    texte = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", texte)
    return re.sub(r"hf_[A-Za-z0-9]{6,}", "hf_***", texte)


def etape(texte: str) -> None:
    print(time.strftime("%H:%M:%S"), texte, flush=True)


def run(commande: str) -> str:
    r = subprocess.run(commande, shell=True, text=True, capture_output=True)
    if r.returncode:
        print(masquer(r.stdout[-3000:]), masquer(r.stderr[-3000:]))
        raise RuntimeError(f"Échec : {masquer(commande)[:160]}")
    return r.stdout


def tourne(motif: str) -> bool:
    return subprocess.run(f"pgrep -f '{motif}'", shell=True, capture_output=True).returncode == 0


def en_arriere_plan(commande: list[str] | str, journal: str, **options) -> subprocess.Popen:
    """Processus détaché : il survit à l'arrêt de ce script (Ctrl+C)."""
    return subprocess.Popen(commande, stdout=open(journal, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True, shell=isinstance(commande, str), **options)


def installer() -> None:
    sudo = "" if os.geteuid() == 0 else "sudo -n "
    if not shutil.which("ffmpeg"):
        etape("Installation de FFmpeg…")
        run(f"{sudo}apt-get -qq update && {sudo}apt-get -qq install -y ffmpeg zstd lshw > /dev/null")
    etape("Dépendances Python (quelques minutes la première fois)…")
    # PyTorch déjà installé avec CUDA : on le fige pour que pip ne le remplace pas par une version CPU
    run(f"{sys.executable} -m pip freeze | grep -iE '^(torch|torchaudio|torchvision)==' > /tmp/contraintes.txt || true")
    lignes = (CODE / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    Path("/tmp/requirements-base.txt").write_text(
        "\n".join(l for l in lignes if not l.strip().startswith(FACULTATIFS)), encoding="utf-8")
    run(f"{sys.executable} -m pip install -q -c /tmp/contraintes.txt -r /tmp/requirements-base.txt")
    for outil in ("torchcodec",) + FACULTATIFS:
        r = subprocess.run(f"{sys.executable} -m pip install -q -c /tmp/contraintes.txt {outil}",
                           shell=True, text=True, capture_output=True)
        print(f"  {outil} :", "installé" if r.returncode == 0 else "non installé (le moteur s'en passera)")

    # Chatterbox (voix expressive) et MuseTalk (synchronisation labiale) : environnements à part, en arrière-plan
    chatterbox = OUTILS / "chatterbox"
    if not (chatterbox / ".pret").exists() and not tourne("chatterbox[-]tts"):
        en_arriere_plan(f"{sys.executable} -m pip install -q uv && uv venv -q --allow-existing -p 3.11 {chatterbox}"
                        f" && uv pip install -q -p {chatterbox}/bin/python chatterbox-tts 'setuptools<81'"
                        f" && touch {chatterbox}/.pret", "/tmp/chatterbox.log")
        etape("Chatterbox : installation lancée en arrière-plan (≈ 5 min, journal /tmp/chatterbox.log)")
    if not (OUTILS / "musetalk" / "code" / ".pret").exists() and not tourne("installer[_]musetalk"):
        en_arriere_plan(["bash", str(CODE / "kaggle" / "installer_musetalk.sh")], "/tmp/musetalk.log",
                        env=os.environ | {"MUSETALK_RACINE": str(OUTILS / "musetalk")})
        etape("MuseTalk : installation lancée en arrière-plan (≈ 5 min, journal /tmp/musetalk.log)")

    if not shutil.which("ollama"):
        etape("Installation d'Ollama…")
        run(f"curl -fsSL https://ollama.com/install.sh | {sudo}sh")
    if not tourne("ollama serve"):
        en_arriere_plan(["ollama", "serve"], "/tmp/ollama.log",
                        env=os.environ | {"OLLAMA_MODELS": str(MODELES / "ollama")})
        time.sleep(5)
    for modele in (MODELE_TRADUCTION, MODELE_VISION):
        etape(f"Modèle {modele} (téléchargé une seule fois)…")
        run(f"OLLAMA_MODELS={MODELES / 'ollama'} ollama pull {modele}")


def cle_acces() -> str:
    """Clé fixée par VOXLINGO_CLE, sinon tirée au hasard une fois et gardée : le lien reste valable."""
    if os.environ.get("VOXLINGO_CLE"):
        return os.environ["VOXLINGO_CLE"]
    fichier = DONNEES / ".cle"
    if not fichier.is_file():
        DONNEES.mkdir(parents=True, exist_ok=True)
        fichier.write_text(secrets.token_urlsafe(18), encoding="utf-8")
        fichier.chmod(0o600)
    return fichier.read_text(encoding="utf-8").strip()


def demarrer_moteur(cle: str) -> subprocess.Popen | None:
    import httpx

    libs = []  # bibliothèques CUDA fournies par pip (cuBLAS, cuDNN), pour faster-whisper
    for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            libs.append(list(__import__(module, fromlist=["_"]).__path__)[0])
        except Exception:
            pass
    env = os.environ | {
        "CLE_ACCES": cle, "DATA_DIR": str(DONNEES), "MODELS_DIR": str(MODELES),
        "OLLAMA_MODEL": MODELE_TRADUCTION, "MODELE_VISION": MODELE_VISION, "COQUI_TOS_AGREED": "1",
        "MPLBACKEND": "Agg", "CHATTERBOX_PYTHON": str(OUTILS / "chatterbox" / "bin" / "python"),
        "MUSETALK_PYTHON": str(OUTILS / "musetalk" / "env" / "bin" / "python"),
        "MUSETALK_DIR": str(OUTILS / "musetalk" / "code"),
        "LD_LIBRARY_PATH": ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
    }
    if env.get("HF_TOKEN_ECRITURE"):
        r = subprocess.run([sys.executable, "-m", "sauvegarde", "restaurer"], cwd=CODE / "backend", text=True,
                           capture_output=True, env=env | {"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        etape("Sauvegarde : " + masquer((r.stdout.strip() or r.stderr.strip()[-1000:]) or "aucune réponse"))
    try:  # relance pendant un doublage : on garde le moteur en marche plutôt que d'interrompre le traitement
        occupes = [j["id"] for j in httpx.get("http://127.0.0.1:8000/jobs", headers={"X-Cle-Acces": cle},
                                              timeout=5).json() if j["etat"] in ("en_cours", "en_attente")]
    except Exception:
        occupes = []
    if occupes:
        etape(f"Doublage en cours ({', '.join(occupes)}) : le moteur déjà lancé est gardé tel quel.")
        return None
    subprocess.run("pkill -f 'uvicorn main:app'", shell=True)  # relance : l'ancien moteur laisse la place
    time.sleep(2)
    moteur = en_arriere_plan([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
                             "/tmp/voxlingo.log", cwd=CODE / "backend", env=env)
    for _ in range(120):
        try:
            sante = httpx.get("http://127.0.0.1:8000/health", headers={"X-Cle-Acces": cle}, timeout=5).json()
            etape(f"Moteur prêt : {sante['device']}, {sante['vram_totale_go']} Go de VRAM · {sante['modeles']}")
            return moteur
        except Exception:
            if moteur.poll() is not None:
                print(open("/tmp/voxlingo.log").read()[-3000:])
                raise RuntimeError("Le moteur s'est arrêté au démarrage (journal ci-dessus).")
            time.sleep(2)
    raise RuntimeError("Le moteur ne répond pas (journal : /tmp/voxlingo.log).")


def tunnel() -> str:
    """Tunnel Cloudflare gratuit ; un tunnel déjà lancé est réutilisé (l'adresse ne change pas)."""
    journal = Path("/tmp/tunnel.log")
    if not tourne("cloudflared tunnel"):
        binaire = OUTILS / "cloudflared"
        if not binaire.exists():
            OUTILS.mkdir(parents=True, exist_ok=True)
            run(f"wget -q -O {binaire} https://github.com/cloudflare/cloudflared/releases/latest/download/"
                f"cloudflared-linux-amd64 && chmod +x {binaire}")
        journal.write_text("")
        en_arriere_plan([str(binaire), "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"], str(journal))
    for _ in range(60):
        trouve = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", journal.read_text(errors="replace"))
        if trouve:
            return trouve.group(0)
        time.sleep(2)
    raise RuntimeError("Le tunnel n'a pas démarré (journal : /tmp/tunnel.log).")


def lien_de_connexion(adresse: str, cle: str) -> str:
    return f"{ADRESSE_VERCEL.rstrip('/')}/?moteur={quote(adresse, safe='')}&cle={quote(cle, safe='')}#/"


def lancer() -> tuple[subprocess.Popen | None, str, str]:
    """Installe ce qui manque, démarre le moteur et le tunnel. Renvoie (moteur, adresse, clé)."""
    etape(f"Code : {CODE} · données : {DONNEES} · modèles : {MODELES}")
    installer()
    cle = cle_acces()
    moteur = demarrer_moteur(cle)
    return moteur, tunnel(), cle


def suivre(moteur: subprocess.Popen | None) -> None:
    """Affiche l'état du moteur chaque minute (garde aussi une séance Kaggle/Colab active)."""
    try:
        while True:
            time.sleep(60)
            en_marche = moteur.poll() is None if moteur is not None else tourne("uvicorn main:app")
            etat = "en marche" if en_marche else "ARRÊTÉ"
            derniere = (open("/tmp/voxlingo.log", errors="replace").read().strip().splitlines() or [""])[-1]
            print(time.strftime("%H:%M"), "· moteur", etat, "·", masquer(derniere)[-150:], flush=True)
    except KeyboardInterrupt:
        print("Suivi arrêté (le moteur et le tunnel restent lancés).")


def main() -> None:
    moteur, adresse, cle = lancer()
    print(f"\n✅ Moteur en ligne : {adresse}\n👉 Ouvrez ce lien (il contient la clé, gardez-le pour vous) :\n"
          f"{lien_de_connexion(adresse, cle)}\n", flush=True)
    suivre(moteur)


if __name__ == "__main__":
    main()
