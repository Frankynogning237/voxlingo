"""Serveur local FastAPI de VoxLingo (http://localhost:8000)."""

from __future__ import annotations

import hmac
import json
import logging
import uuid
import platform
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Literal
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import jobs
import sauvegarde
from config import get_runtime, get_settings
from pipeline.extract import make_filmstrip
from pipeline.segments import est_trop_long, marquer_fin_max

_LOG_DIR = get_settings().data_dir / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "voxlingo.log", encoding="utf-8"),
    ],
)
# Les requêtes HTTP vers Hugging Face et Ollama encombrent les journaux
for _bruyant in ("httpx", "httpx2", "huggingface_hub"):
    logging.getLogger(_bruyant).setLevel(logging.WARNING)
logger = logging.getLogger("voxlingo")

app = FastAPI(title="VoxLingo", version="0.1.0")



@app.middleware("http")
async def verifier_cle(request: Request, call_next):
    """Moteur exposé sur Internet : exige la clé d'accès (en-tête, ou paramètre `cle` pour les
    vidéos, images et téléchargements que le navigateur charge lui-même)."""
    cle = get_settings().cle_acces
    if cle and request.method != "OPTIONS":
        fournie = request.headers.get("x-cle-acces") or request.query_params.get("cle") or ""
        if not hmac.compare_digest(fournie.encode(), cle.encode()):
            return JSONResponse({"detail": "Clé d'accès manquante ou incorrecte."}, status_code=401)
    return await call_next(request)


# Ajouté après la vérification de la clé : il l'enveloppe, donc même un refus porte les en-têtes CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origines,
    allow_origin_regex=get_settings().cors_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _first_line(cmd: list[str]) -> str | None:
    """Exécute une commande et renvoie la première ligne de sa sortie."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else None


@lru_cache
def tool_versions() -> dict[str, str | None]:
    """Versions des outils externes utilisés par le pipeline."""
    try:
        import torch

        torch_version: str | None = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "ffmpeg": _first_line(["ffmpeg", "-version"]),
        "ollama": _first_line(["ollama", "--version"]),
    }


@app.on_event("startup")
def startup() -> None:
    runtime = get_runtime()
    logger.info("Démarrage : device=%s, mode=%s", runtime.device, runtime.mode)
    # Une tâche "en cours" au démarrage a été interrompue par l'arrêt du serveur
    for status in jobs.list_jobs():
        if status["etat"] in ("en_attente", "en_cours"):
            jobs.update_status(
                status["id"], etat="erreur",
                erreur="Traitement interrompu par l'arrêt du serveur.",
            )


@app.on_event("shutdown")
def shutdown() -> None:
    sauvegarde.a_l_arret()  # Kaggle/Colab : dernière sauvegarde avant la fin de la séance


# Formats lus par FFmpeg. Aucune limite de taille ni de durée : seul l'espace disque compte.
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}
CHUNK_BYTES = 8 * 1024**2
ESPACE_MINIMAL = 2 * 1024**3  # garder au moins 2 Go libres pour l'audio et les fichiers générés


def _espace_libre() -> int:
    return shutil.disk_usage(get_settings().projects_dir).free


def _fichier_televersement(televersement_id: str) -> Path:
    if not televersement_id.isalnum():
        raise HTTPException(404, "Envoi introuvable.")
    dossier = get_settings().data_dir / "televersements"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier / f"{televersement_id}.part"


@app.post("/televersements", status_code=201)
def debut_televersement() -> dict:
    """Commence l'envoi d'une vidéo par morceaux (les tunnels gratuits limitent chaque requête à ~100 Mo)."""
    televersement_id = uuid.uuid4().hex
    _fichier_televersement(televersement_id).touch()
    return {"id": televersement_id}


@app.put("/televersements/{televersement_id}")
async def morceau_televersement(televersement_id: str, request: Request, debut: int = 0) -> dict:
    """Ajoute un morceau à la fin du fichier ; `debut` (position attendue) rend les nouvelles tentatives sûres."""
    fichier = _fichier_televersement(televersement_id)
    if not fichier.is_file():
        raise HTTPException(404, "Envoi introuvable.")
    taille = fichier.stat().st_size
    if debut != taille:
        if debut + int(request.headers.get("content-length") or 0) == taille:
            return {"taille": taille}  # morceau déjà reçu : nouvelle tentative après une coupure
        raise HTTPException(409, f"Morceau inattendu : {taille} octets déjà reçus.")
    if _espace_libre() < ESPACE_MINIMAL:
        raise HTTPException(507, "Espace disque insuffisant pour cette vidéo.")
    with fichier.open("ab") as out:
        try:
            async for bloc in request.stream():
                out.write(bloc)
        except BaseException:
            out.truncate(taille)  # morceau coupé en route : on l'efface pour qu'il soit renvoyé en entier
            raise
    return {"taille": fichier.stat().st_size}


async def _enregistrer_video(video: UploadFile, destination: Path) -> int:
    """Écrit la vidéo par morceaux pour ne jamais la charger entière en mémoire ; renvoie sa taille."""
    taille = 0
    with destination.open("wb") as out:
        while chunk := await video.read(CHUNK_BYTES):
            taille += len(chunk)
            # Vérification régulière (tous les ~256 Mo) de l'espace disque restant
            if taille % (32 * CHUNK_BYTES) < len(chunk) and _espace_libre() < ESPACE_MINIMAL:
                raise HTTPException(
                    507,
                    "Espace disque insuffisant pour cette vidéo : libérez de la place "
                    f"(au moins {ESPACE_MINIMAL // 1024**3} Go doivent rester libres).",
                )
            out.write(chunk)
    return taille


@app.post("/jobs", status_code=201)
async def create_job(
    video: UploadFile | None = File(None),
    televersement: str | None = Form(None),
    nom_fichier: str | None = Form(None),
    langue_cible: str = Form(...),
    conserver_fond: bool = Form(True),
    synchro_labiale: bool = Form(False),
    sous_titres: bool = Form(True),
    volume_fond: int = Form(100, ge=0, le=100),
    ajuster_trop_longs: bool = Form(True),
    strategie_duree: Literal["reformuler", "accelerer"] = Form("reformuler"),
    voix_bibliotheque: bool = Form(False),
    intonation_originale: bool = Form(True),
    glossaire: str = Form("", max_length=5000),
    consignes: str = Form("", max_length=2000),
    debut_s: float | None = Form(None, ge=0),
    fin_s: float | None = Form(None, gt=0),
) -> dict:
    """Reçoit une vidéo et la langue cible, enregistre le projet et lance le pipeline."""
    # La vidéo arrive dans la requête (`video`) ou a déjà été envoyée par morceaux (`televersement`)
    if video is None and not televersement:
        raise HTTPException(400, "Aucune vidéo reçue.")
    nom = Path((video.filename if video else nom_fichier) or "video").name
    ext = Path(nom).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, "Format non pris en charge : utilisez MP4, MOV, MKV, WEBM, AVI, M4V, WMV, FLV, TS ou MPG."
        )
    if debut_s is not None and fin_s is not None and fin_s - debut_s < 1:
        raise HTTPException(400, "Le passage à garder doit durer au moins 1 seconde.")
    if langue_cible not in get_settings().target_languages:
        raise HTTPException(400, f"Langue cible non prise en charge : {langue_cible}.")

    job_id = jobs.new_job_id()
    folder = jobs.project_dir(job_id)
    folder.mkdir(parents=True)
    destination = folder / f"input{ext}"

    try:
        if video is not None:
            taille = await _enregistrer_video(video, destination)
        else:
            partiel = _fichier_televersement(televersement or "")
            if not partiel.is_file():
                raise HTTPException(404, "Envoi introuvable : renvoyez la vidéo.")
            taille = partiel.stat().st_size
            shutil.move(partiel, destination)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    logger.info("Vidéo reçue : %s (%.1f Mo) → tâche %s", nom, taille / 1024**2, job_id)
    options = {
        "conserver_fond": conserver_fond,
        "synchro_labiale": synchro_labiale,
        "sous_titres": sous_titres,
        "volume_fond": volume_fond,
        "ajuster_trop_longs": ajuster_trop_longs,
        "strategie_duree": strategie_duree,
        "voix_bibliotheque": voix_bibliotheque,
        "intonation_originale": intonation_originale,
        "glossaire": glossaire.strip(),
        "consignes": consignes.strip(),
    }
    if debut_s is not None or fin_s is not None:
        # Raccourcir la vidéo : seul ce passage sera traité (coupe faite au début du traitement)
        from pipeline.extract import probe_duration

        duree = probe_duration(destination)
        debut, fin = max(0.0, debut_s or 0.0), min(duree, fin_s if fin_s is not None else duree)
        if fin - debut < 1:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, "Le passage à garder doit durer au moins 1 seconde.")
        if debut > 0.05 or fin < duree - 0.05:
            options["decoupe"] = {"debut_s": round(debut, 2), "fin_s": round(fin, 2)}
    status = jobs.create_job(job_id, nom, destination.name, langue_cible, options)
    jobs.start_job(job_id)
    return status


@app.get("/jobs")
def list_jobs() -> list[dict]:
    """Historique des tâches."""
    return jobs.list_jobs()


def _get_status_or_404(job_id: str) -> dict:
    status = jobs.read_status(job_id) if job_id.isalnum() else None
    if status is None:
        raise HTTPException(404, "Projet introuvable.")
    return status


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Statut d'une tâche : état, étape courante, progression, erreurs."""
    return _get_status_or_404(job_id)


@app.get("/jobs/{job_id}/segments")
def get_segments(job_id: str) -> dict:
    """Transcription et traductions, segment par segment."""
    _get_status_or_404(job_id)
    status = _get_status_or_404(job_id)
    if not (jobs.project_dir(job_id) / "segments.json").is_file():
        raise HTTPException(404, "La transcription n'est pas encore disponible.")
    data = jobs.read_segments(job_id)
    if data["segments"] and "fin_max" not in data["segments"][0]:
        marquer_fin_max(data["segments"], status.get("duree_s"))  # projets créés avant l'ajustement
    from pipeline.voice import fichier_brut

    dossier_voix = jobs.project_dir(job_id) / "voix"
    for seg in data["segments"]:
        seg["voix_disponible"] = fichier_brut(dossier_voix, seg["id"]).is_file()
    return data


@app.get("/jobs/{job_id}/segments/{seg_id}/voix")
def segment_voice(job_id: str, seg_id: int) -> FileResponse:
    """Voix doublée d'une phrase (dernière génération), pour l'écouter seule."""
    from pipeline.voice import fichier_brut

    _get_status_or_404(job_id)
    chemin = fichier_brut(jobs.project_dir(job_id) / "voix", seg_id)
    if not chemin.is_file():
        raise HTTPException(404, "La voix de cette phrase n'a pas encore été générée.")
    return FileResponse(chemin, media_type="audio/wav")


@app.post("/jobs/{job_id}/segments/{seg_id}/refaire")
def redo_segment_voice(job_id: str, seg_id: int) -> dict:
    """Refait la voix d'une seule phrase (nouvelle prise), puis réassemble la vidéo.
    Les autres phrases, déjà à jour, sont gardées telles quelles."""
    import json as _json

    from pipeline.voice import fichier_brut

    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Un traitement est en cours : attendez sa fin.")
    data = jobs.read_segments(job_id)
    if not any(s["id"] == seg_id for s in data["segments"]):
        raise HTTPException(404, "Phrase introuvable.")
    dossier_voix = jobs.project_dir(job_id) / "voix"
    fichier_brut(dossier_voix, seg_id).unlink(missing_ok=True)
    manifeste = dossier_voix / "manifeste.json"
    if manifeste.is_file():
        entrees = _json.loads(manifeste.read_text(encoding="utf-8"))
        entrees.pop(str(seg_id), None)
        manifeste.write_text(_json.dumps(entrees, ensure_ascii=False, indent=1), encoding="utf-8")
    status = jobs.update_status(job_id, traduction_validee=True, etat="en_attente")
    jobs.start_voices(job_id)
    return status


class TraductionModifiee(BaseModel):
    id: int
    traduction: str
    texte: str | None = None  # correction de la transcription d'origine


class EditionTraductions(BaseModel):
    segments: list[TraductionModifiee]


def _require_editable(job_id: str) -> dict:
    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Un traitement est en cours : attendez sa fin pour modifier la traduction.")
    if not status.get("moteur_traduction"):
        raise HTTPException(409, "La traduction n'a pas encore été générée.")
    return status


@app.put("/jobs/{job_id}/segments")
def save_translations(job_id: str, edition: EditionTraductions) -> dict:
    """Enregistre les traductions modifiées (brouillon) et recalcule les alertes « Trop long »."""
    status = _require_editable(job_id)
    data = jobs.read_segments(job_id)
    marquer_fin_max(data["segments"], status.get("duree_s"))
    par_id = {e.id: e.traduction.strip() for e in edition.segments}
    textes = {e.id: e.texte.strip() for e in edition.segments if e.texte and e.texte.strip()}
    for seg in data["segments"]:
        if seg["id"] in par_id:
            if not par_id[seg["id"]]:
                raise HTTPException(400, f"La traduction du segment {seg['id'] + 1} est vide.")
            seg["traduction"] = par_id[seg["id"]]
            correction = textes.get(seg["id"])
            if correction and correction != seg["texte"]:
                # Transcription corrigée : la traduction devra être refaite ; le minutage des mots
                # ne correspond plus au nouveau texte (le découpage se fera sur les pauses)
                seg["texte"], seg["texte_corrige"], seg["a_retraduire"] = correction, True, True
                seg["mots"] = []
            seg["trop_long"] = est_trop_long(seg, seg["traduction"], status["langue_source"], status["langue_cible"],
                                             jobs.strategie(status))
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False, brouillon_le=jobs._now())
    return data


@app.post("/jobs/{job_id}/reinitialiser")
def reset_translations(job_id: str) -> dict:
    """Revient aux traductions proposées par l'IA."""
    status = _require_editable(job_id)
    data = jobs.read_segments(job_id)
    marquer_fin_max(data["segments"], status.get("duree_s"))
    for seg in data["segments"]:
        seg["traduction"] = seg["traduction_ia"]
        seg["trop_long"] = est_trop_long(seg, seg["traduction"], status["langue_source"], status["langue_cible"],
                                         jobs.strategie(status))
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False)
    return data


class ChoixStrategie(BaseModel):
    strategie_duree: Literal["reformuler", "accelerer"] | None = None
    glossaire: str | None = Field(None, max_length=5000)
    consignes: str | None = Field(None, max_length=2000)


@app.put("/jobs/{job_id}/options")
def set_strategy(job_id: str, choix: ChoixStrategie) -> dict:
    """Change la stratégie face aux phrases trop longues (et recalcule les alertes), ou le glossaire
    et les consignes de traduction (appliqués à la prochaine traduction)."""
    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Un traitement est en cours : attendez sa fin pour changer cette option.")
    options = {**(status.get("options") or {}),
               **{k: (v.strip() if isinstance(v, str) else v) for k, v in choix.model_dump(exclude_none=True).items()}}
    status = jobs.update_status(job_id, options=options)
    if choix.strategie_duree is None:
        return status
    if (jobs.project_dir(job_id) / "segments.json").is_file() and status.get("moteur_traduction"):
        data = jobs.read_segments(job_id)
        marquer_fin_max(data["segments"], status.get("duree_s"))
        for seg in data["segments"]:
            seg["trop_long"] = est_trop_long(seg, seg["traduction"], status["langue_source"], status["langue_cible"],
                                             choix.strategie_duree)
        jobs.write_segments(job_id, data)
    return status


class DemandeTraduction(BaseModel):
    ids: list[int] | None = None  # None : toutes les phrases ; sinon seulement celles-ci


@app.post("/jobs/{job_id}/traduire")
def retranslate(job_id: str, demande: DemandeTraduction | None = None) -> dict:
    """Relance la traduction par l'IA : toutes les phrases (écrase les modifications), ou seulement
    les phrases indiquées (texte d'origine corrigé), les autres traductions étant conservées."""
    _require_editable(job_id)
    ids = set(demande.ids) if demande and demande.ids is not None else None
    if ids is not None and not ids:
        raise HTTPException(400, "Aucune phrase à retraduire.")
    status = jobs.update_status(job_id, etat="en_attente")
    jobs.start_translation(job_id, ids)
    return status


class DemandeAjustement(BaseModel):
    ids: list[int] | None = None  # None = toutes les phrases trop longues


@app.post("/jobs/{job_id}/ajuster")
def adjust_translations(job_id: str, demande: DemandeAjustement) -> dict:
    """Ajuste automatiquement les phrases trop longues : silence suivant, raccourcissement, découpage."""
    _require_editable(job_id)
    status = jobs.update_status(job_id, etat="en_attente")
    jobs.start_adjustment(job_id, set(demande.ids) if demande.ids is not None else None)
    return status


@app.post("/jobs/{job_id}/valider")
def validate_translations(job_id: str, edition: EditionTraductions) -> dict:
    """Enregistre et valide les traductions, puis lance la génération des voix et l'export.

    Sert aussi après une correction : seules les phrases dont le texte a changé sont
    régénérées (cache des voix), puis la vidéo est réassemblée.
    """
    save_translations(job_id, edition)
    status = jobs.update_status(job_id, traduction_validee=True, etat="en_attente", valide_le=jobs._now())
    jobs.start_voices(job_id)
    return status


def _fichier_projet(job_id: str, nom: str, message: str) -> Path:
    path = jobs.project_dir(job_id) / nom
    if not path.is_file():
        raise HTTPException(404, message)
    return path


def _nom_telechargement(status: dict, extension: str) -> str:
    return f"{Path(status['nom_fichier']).stem}_{status['langue_cible']}.{extension}"


@app.get("/jobs/{job_id}/download")
def download_video(job_id: str) -> FileResponse:
    """Vidéo doublée, au format exact de la vidéo fournie."""
    import mimetypes

    status = _get_status_or_404(job_id)
    finale = _video_finale(job_id, status)
    type_mime = mimetypes.guess_type(finale.name)[0] or "application/octet-stream"
    return FileResponse(finale, media_type=type_mime,
                        filename=_nom_telechargement(status, finale.suffix.lstrip(".").lower()))


@app.get("/jobs/{job_id}/dubbed")
def stream_dubbed(job_id: str) -> FileResponse:
    """Vidéo doublée pour la lecture dans l'interface (copie MP4 si le format d'origine ne s'y lit pas)."""
    import mimetypes

    status = _get_status_or_404(job_id)
    apercu = _apercu_finale(job_id, status)
    return FileResponse(apercu, media_type=mimetypes.guess_type(apercu.name)[0] or "video/mp4")


RACINE = Path(__file__).resolve().parent.parent


@app.get("/jobs/{job_id}/kaggle")
def kaggle_package(job_id: str) -> FileResponse:
    """Paquet pour la synchronisation labiale sur Kaggle : vidéo d'origine, piste doublée, carnet, guide."""
    import zipfile

    status = _get_status_or_404(job_id)
    dossier = jobs.project_dir(job_id)
    piste = _fichier_projet(job_id, "doublage.wav", "Générez d'abord les voix du projet.")
    paquet = dossier / f"voxlingo_kaggle_{job_id}.zip"
    guide = (
        "VoxLingo - synchronisation labiale sur Kaggle (gratuit, environ 30 h de GPU par semaine)\n\n"
        "1. Ouvrez https://www.kaggle.com/code et créez un carnet : File > Import Notebook > lipsync_kaggle.ipynb\n"
        "2. Session options : Accelerator = GPU T4 x2 (ou P100), Internet = activé\n"
        "3. Add Input > Upload : ce fichier .zip\n"
        "4. Run All, puis téléchargez video_synchronisee.mp4 (Output)\n"
        "5. Dans VoxLingo, écran Résultat > Synchronisation labiale > Importer la vidéo synchronisée\n"
    )
    doublee = _fichier_projet(job_id, status.get("fichier_sortie") or "doublage.mp4",
                              "Générez d'abord les voix du projet.")
    if doublee.suffix.lower() != ".mp4":
        # Le carnet de synchronisation lit du MP4 ; le résultat sera remis au format d'origine à l'import
        from pipeline.format import apercu_navigateur

        doublee = apercu_navigateur(doublee, dossier / "pour_kaggle.mp4", toujours=True) or doublee
    with zipfile.ZipFile(paquet, "w", zipfile.ZIP_STORED) as z:
        z.write(doublee, "video_originale.mp4")
        z.write(piste, "voix_doublee.wav")
        z.write(RACINE / "kaggle" / "lipsync_kaggle.ipynb", "lipsync_kaggle.ipynb")
        z.writestr("LISEZMOI.txt", guide)
    return FileResponse(paquet, media_type="application/zip", filename=paquet.name)


@app.post("/jobs/{job_id}/synchro")
async def import_lipsync(job_id: str, video: UploadFile = File(...)) -> dict:
    """Importe la vidéo synchronisée par Kaggle : elle devient la vidéo doublée téléchargeable."""
    from pipeline.extract import probe_duration

    from pipeline.format import apercu_navigateur, conformer

    status = _get_status_or_404(job_id)
    dossier = jobs.project_dir(job_id)
    recue = dossier / "synchro_recue.mp4"
    with recue.open("wb") as sortie:
        while chunk := await video.read(CHUNK_BYTES):
            sortie.write(chunk)
    try:
        duree = probe_duration(recue)
    except Exception:
        recue.unlink(missing_ok=True)
        raise HTTPException(400, "Ce fichier n'est pas une vidéo lisible.") from None
    if status.get("duree_s") and abs(duree - status["duree_s"]) > 2:
        recue.unlink(missing_ok=True)
        raise HTTPException(400, f"Cette vidéo dure {duree:.0f} s au lieu de {status['duree_s']:.0f} s : ce n'est pas le bon fichier.")
    # Remise au format exact de la vidéo fournie (résolution, cadence, codecs, conteneur)
    originale = dossier / status["fichier"]
    for ancienne in dossier.glob("doublage_synchro.*"):
        ancienne.unlink()
    cible = conformer(recue, originale, dossier / f"doublage_synchro{originale.suffix.lower()}")
    recue.unlink(missing_ok=True)
    apercu_navigateur(cible, dossier / "apercu_synchro.mp4")
    jobs.update_step(job_id, "synchro_labiale", etat="termine")
    return jobs.update_status(job_id, synchro_importee=True)


@app.delete("/jobs/{job_id}/synchro")
def remove_lipsync(job_id: str) -> dict:
    """Revient à la vidéo doublée sans synchronisation labiale."""
    _get_status_or_404(job_id)
    for fichier in jobs.project_dir(job_id).glob("doublage_synchro.*"):
        fichier.unlink()
    (jobs.project_dir(job_id) / "apercu_synchro.mp4").unlink(missing_ok=True)
    jobs.update_step(job_id, "synchro_labiale", etat="kaggle")
    return jobs.update_status(job_id, synchro_importee=False)


def _video_finale(job_id: str, status: dict) -> Path:
    """Vidéo synchronisée si elle a été importée, sinon la vidéo doublée (au format d'origine)."""
    if status.get("synchro_importee"):
        synchro = next(jobs.project_dir(job_id).glob("doublage_synchro.*"), None)
        if synchro is not None:
            return synchro
    return _fichier_projet(job_id, status.get("fichier_sortie") or "doublage.mp4",
                           "La vidéo doublée n'est pas encore prête.")


def _apercu_finale(job_id: str, status: dict) -> Path:
    """Version lisible par le navigateur de la vidéo finale (elle-même, ou sa copie MP4)."""
    finale = _video_finale(job_id, status)
    copie = finale.with_name("apercu_synchro.mp4" if finale.stem == "doublage_synchro" else "apercu.mp4")
    return copie if copie.is_file() else finale


@app.get("/jobs/{job_id}/subtitles")
def download_subtitles(job_id: str) -> FileResponse:
    """Sous-titres dans la langue cible (SRT)."""
    status = _get_status_or_404(job_id)
    path = _fichier_projet(job_id, "sous-titres.srt", "Les sous-titres n'ont pas été générés pour ce projet.")
    return FileResponse(path, media_type="application/x-subrip", filename=_nom_telechargement(status, "srt"))


@app.get("/jobs/{job_id}/subtitles.vtt")
def subtitles_vtt(job_id: str) -> Response:
    """Sous-titres au format WebVTT, pour l'affichage dans le lecteur de l'interface."""
    _get_status_or_404(job_id)
    srt = _fichier_projet(job_id, "sous-titres.srt", "Les sous-titres n'ont pas été générés pour ce projet.")
    # SRT : 00:00:01,500 ; WebVTT : 00:00:01.500
    corps = re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt.read_text(encoding="utf-8"))
    return Response("WEBVTT\n\n" + corps, media_type="text/vtt; charset=utf-8")


@app.get("/jobs/{job_id}/audio")
def download_audio(job_id: str) -> FileResponse:
    """Piste doublée seule (MP3), créée à la demande."""
    from pipeline.assemble import exporter_mp3

    status = _get_status_or_404(job_id)
    piste = _fichier_projet(job_id, "doublage.wav", "La piste doublée n'est pas encore prête.")
    mp3 = jobs.project_dir(job_id) / "doublage.mp3"
    if not mp3.is_file() or mp3.stat().st_mtime < piste.stat().st_mtime:
        exporter_mp3(piste, mp3)
    return FileResponse(mp3, media_type="audio/mpeg", filename=_nom_telechargement(status, "mp3"))


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """Vidéo d'origine, pour l'aperçu dans l'interface."""
    status = _get_status_or_404(job_id)
    return FileResponse(jobs.project_dir(job_id) / status["fichier"])


@app.get("/jobs/{job_id}/thumbnail")
def get_thumbnail(job_id: str) -> FileResponse:
    """Miniature de la vidéo (générée pendant l'extraction)."""
    _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "thumbnail.jpg"
    if not path.is_file():
        raise HTTPException(404, "Miniature pas encore disponible.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/filmstrip")
def get_filmstrip(job_id: str) -> FileResponse:
    """Bande d'images de la vidéo pour la timeline (créée à la demande pour les anciens projets)."""
    status = _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "filmstrip.jpg"
    if not path.is_file():
        if not status.get("duree_s"):
            raise HTTPException(404, "Bande d'images pas encore disponible.")
        make_filmstrip(jobs.project_dir(job_id) / status["fichier"], path, status["duree_s"])
    if not path.is_file():
        raise HTTPException(500, "Impossible de créer la bande d'images.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/waveform")
def get_waveform(job_id: str) -> list[float]:
    """Amplitudes réelles de l'audio, pour dessiner la forme d'onde."""
    _get_status_or_404(job_id)
    path = jobs.project_dir(job_id) / "waveform.json"
    if not path.is_file():
        raise HTTPException(404, "Forme d'onde pas encore disponible.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/jobs/{job_id}/annuler")
def cancel_job(job_id: str) -> dict:
    """Demande l'arrêt du traitement en cours."""
    status = _get_status_or_404(job_id)
    if status["etat"] not in ("en_attente", "en_cours"):
        raise HTTPException(409, "Ce traitement n'est pas en cours.")
    return jobs.request_cancel(job_id)


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Supprime définitivement un projet et ses fichiers."""
    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Annulez le traitement en cours avant de supprimer ce projet.")
    jobs.delete_job(job_id)
    sauvegarde.apres_suppression(job_id)
    return {"supprime": job_id}


class Duplication(BaseModel):
    langue_cible: str | None = None  # autre langue cible (plusieurs langues d'un clic)


@app.post("/jobs/{job_id}/dupliquer", status_code=201)
def duplicate_job(job_id: str, demande: Duplication | None = None) -> dict:
    """Nouveau projet avec la même vidéo et les mêmes réglages, éventuellement dans une autre langue
    (le traitement démarre dès que les projets précédents sont terminés)."""
    _get_status_or_404(job_id)
    langue = demande.langue_cible if demande else None
    if langue is not None and langue not in get_settings().target_languages:
        raise HTTPException(400, f"Langue cible non prise en charge : {langue}.")
    return jobs.duplicate_job(job_id, langue)


CHOIX_TRANSCRIPTION = [
    {"valeur": "auto", "libelle": "Automatique"},
    {"valeur": "small", "libelle": "Whisper Small (rapide)"},
    {"valeur": "medium", "libelle": "Whisper Medium"},
    {"valeur": "large-v3-turbo", "libelle": "Whisper Large V3 Turbo"},
]
CHOIX_TRADUCTION = [
    {"valeur": "auto", "libelle": "Automatique"},
    {"valeur": "qwen2.5:3b", "libelle": "Qwen 2.5 3B (rapide)"},
    {"valeur": "qwen2.5:7b", "libelle": "Qwen 2.5 7B (meilleure qualité)"},
    {"valeur": "qwen2.5:14b", "libelle": "Qwen 2.5 14B (la meilleure, GPU conseillé)"},
    {"valeur": "nllb", "libelle": "NLLB-200 (sans Ollama)"},
]


class ValeursParametres(BaseModel):
    force_device: Literal["auto", "cpu", "cuda"]
    whisper_model: str
    ollama_model: str
    strategie_duree: Literal["reformuler", "accelerer"]
    sous_titres: bool
    conserver_fond: bool
    volume_fond: int


def _taille_dossier(dossier: Path) -> int:
    return sum(f.stat().st_size for f in dossier.rglob("*") if f.is_file()) if dossier.is_dir() else 0


def _etat_parametres() -> dict:
    import psutil

    from config import PARAMETRES_DEFAUT, lire_parametres

    runtime = get_runtime()
    settings = get_settings()
    memoire = psutil.virtual_memory()
    gpu, vram_utilisee = None, 0.0
    if runtime.device == "cuda":
        import torch

        gpu = torch.cuda.get_device_name(0)
        libre, total = torch.cuda.mem_get_info()
        vram_utilisee = round((total - libre) / 1024**3, 1)
    disque = shutil.disk_usage(settings.projects_dir)
    return {
        "valeurs": lire_parametres(),
        "defauts": PARAMETRES_DEFAUT,
        "choix": {"transcription": CHOIX_TRANSCRIPTION, "traduction": CHOIX_TRADUCTION},
        "materiel": {
            "device": runtime.device,
            "gpu": gpu,
            "processeur": platform.processor() or platform.machine(),
            "vram_totale_go": runtime.vram_go,
            "vram_utilisee_go": vram_utilisee,
            "ram_totale_go": round(memoire.total / 1024**3, 1),
            "ram_utilisee_go": round((memoire.total - memoire.available) / 1024**3, 1),
            "ram_libre_go": round(memoire.available / 1024**3, 1),
        },
        "stockage": {
            "dossier": str(settings.projects_dir),
            "projets_octets": _taille_dossier(settings.projects_dir),
            "modeles_octets": _taille_dossier(settings.models_dir),
            "disque_total_octets": disque.total,
            "disque_libre_octets": disque.free,
        },
    }


@app.get("/parametres")
def get_parametres() -> dict:
    """Réglages, matériel détecté et espace de stockage, pour la page Paramètres."""
    return _etat_parametres()


@app.put("/parametres")
def set_parametres(valeurs: ValeursParametres) -> dict:
    """Enregistre les réglages (data/parametres.json) ; ils s'appliquent aux prochains traitements."""
    from config import ecrire_parametres

    if valeurs.whisper_model not in {c["valeur"] for c in CHOIX_TRANSCRIPTION}:
        raise HTTPException(400, "Modèle de transcription inconnu.")
    if valeurs.ollama_model not in {c["valeur"] for c in CHOIX_TRADUCTION}:
        raise HTTPException(400, "Modèle de traduction inconnu.")
    ecrire_parametres({**valeurs.model_dump(), "volume_fond": max(0, min(100, valeurs.volume_fond))})
    tool_versions.cache_clear()
    return _etat_parametres()


@app.post("/parametres/ouvrir-dossier")
def ouvrir_dossier() -> dict:
    """Ouvre le dossier des projets dans l'Explorateur Windows."""
    import os
    import sys

    if sys.platform != "win32":
        raise HTTPException(400, "Le dossier des projets ne peut s'ouvrir que sur un moteur installé sur ce PC Windows.")
    os.startfile(get_settings().projects_dir)  # type: ignore[attr-defined]  # Windows uniquement
    return {"ok": True}


# ───────────────────────── Locuteurs (phase 6) ─────────────────────────

def _segments_modifiables(job_id: str) -> tuple[dict, dict]:
    status = _get_status_or_404(job_id)
    if status["etat"] in ("en_attente", "en_cours"):
        raise HTTPException(409, "Un traitement est en cours : attendez sa fin pour modifier les locuteurs.")
    if not (jobs.project_dir(job_id) / "segments.json").is_file():
        raise HTTPException(404, "La transcription n'est pas encore disponible.")
    return status, jobs.read_segments(job_id)


def _fiche_locuteurs(data: dict) -> list[dict]:
    from pipeline.voice_bank import registre, sexe_estime

    total = sum(s["fin"] - s["debut"] for s in data["segments"]) or 1.0
    fiche = []
    for code, infos in jobs.locuteurs(data).items():
        phrases = [s for s in data["segments"] if jobs.locuteur(s) == code]
        duree = sum(s["fin"] - s["debut"] for s in phrases)
        sexe = sexe_estime(infos.get("f0_hz"))
        fiche.append({
            "code": code, **infos,
            "sexe_estime": sexe, "registre": registre(sexe or "homme", infos.get("f0_hz")),
            "nb_phrases": len(phrases), "duree_s": round(duree, 1), "part": round(100 * duree / total),
        })
    return fiche


@app.get("/jobs/{job_id}/locuteurs")
def get_speakers(job_id: str) -> list[dict]:
    """Locuteurs du projet : nom, couleur, voix choisie, temps de parole, timbre mesuré."""
    _get_status_or_404(job_id)
    if not (jobs.project_dir(job_id) / "segments.json").is_file():
        return []
    return _fiche_locuteurs(jobs.read_segments(job_id))


class ModifLocuteur(BaseModel):
    nom: str | None = None
    voix: str | None = None  # « clone » ou identifiant d'une voix de la bibliothèque
    timbre: dict[str, float] | None = None  # réglages de timbre (voir voice_bank.TIMBRE_BORNES)


@app.put("/jobs/{job_id}/locuteurs/{code}")
def update_speaker(job_id: str, code: str, modif: ModifLocuteur) -> list[dict]:
    """Renomme un locuteur ou change sa voix (seules ses phrases seront régénérées)."""
    from pipeline import voice_bank

    _, data = _segments_modifiables(job_id)
    fiche = jobs.locuteurs(data)
    if code not in fiche:
        raise HTTPException(404, "Locuteur introuvable.")
    if modif.nom is not None:
        fiche[code]["nom"] = modif.nom.strip() or fiche[code]["nom"]
    if modif.voix is not None:
        if modif.voix != "clone" and voice_bank.trouver(modif.voix) is None:
            raise HTTPException(400, "Voix inconnue dans la bibliothèque.")
        fiche[code]["voix"] = modif.voix
    if modif.timbre is not None:
        fiche[code]["timbre"] = voice_bank.nettoyer_timbre(modif.timbre)
    data["locuteurs"] = fiche
    jobs.write_segments(job_id, data)
    return _fiche_locuteurs(data)


class Fusion(BaseModel):
    source: str
    cible: str


@app.post("/jobs/{job_id}/locuteurs/fusion")
def merge_speakers(job_id: str, fusion: Fusion) -> list[dict]:
    """Fusionne deux locuteurs détectés à tort comme différents : `source` devient `cible`."""
    _, data = _segments_modifiables(job_id)
    fiche = jobs.locuteurs(data)
    if fusion.source not in fiche or fusion.cible not in fiche or fusion.source == fusion.cible:
        raise HTTPException(400, "Choisissez deux locuteurs différents.")
    for seg in data["segments"]:
        if jobs.locuteur(seg) == fusion.source:
            seg["locuteur"] = fusion.cible
    del fiche[fusion.source]
    data["locuteurs"] = fiche
    # L'échantillon de voix de la cible doit intégrer les nouvelles phrases
    (jobs.project_dir(job_id) / "voix" / f"reference_{fusion.cible}.wav").unlink(missing_ok=True)
    jobs.write_segments(job_id, data)
    return _fiche_locuteurs(data)


class ChoixLocuteur(BaseModel):
    locuteur: str


@app.put("/jobs/{job_id}/segments/{seg_id}/locuteur")
def set_segment_speaker(job_id: str, seg_id: int, choix: ChoixLocuteur) -> dict:
    """Change le locuteur d'une phrase (correction d'une erreur de détection)."""
    _, data = _segments_modifiables(job_id)
    if choix.locuteur not in jobs.locuteurs(data):
        raise HTTPException(400, "Locuteur inconnu.")
    seg = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "Phrase introuvable.")
    ancien = jobs.locuteur(seg)
    seg["locuteur"] = choix.locuteur
    for code in (ancien, choix.locuteur):
        (jobs.project_dir(job_id) / "voix" / f"reference_{code}.wav").unlink(missing_ok=True)
    jobs.write_segments(job_id, data)
    return data


class ChoixTon(BaseModel):
    ton: Literal["energique", "calme", "intense", "doux", "neutre"]


@app.put("/jobs/{job_id}/segments/{seg_id}/ton")
def set_segment_tone(job_id: str, seg_id: int, choix: ChoixTon) -> dict:
    """Corrige le ton d'une phrase : il règle l'expressivité de la voix doublée."""
    _, data = _segments_modifiables(job_id)
    seg = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "Phrase introuvable.")
    seg["ton"], seg["ton_corrige"] = choix.ton, True
    jobs.write_segments(job_id, data)
    return data


@app.post("/jobs/{job_id}/locuteurs/detecter")
def detect_speakers(job_id: str) -> dict:
    """(Re)lance l'identification des locuteurs d'un projet déjà transcrit."""
    _segments_modifiables(job_id)
    status = jobs.update_status(job_id, etat="en_attente")
    jobs.start_speakers(job_id)
    return status


@app.post("/jobs/{job_id}/locuteurs/attribuer")
def auto_assign_voices(job_id: str) -> list[dict]:
    """Attribue à chaque locuteur une voix différente de la bibliothèque, selon sa voix d'origine."""
    from pipeline import voice_bank

    _, data = _segments_modifiables(job_id)
    fiche = jobs.locuteurs(data)
    for code, voix_id in voice_bank.attribuer(fiche).items():
        fiche[code]["voix"] = voix_id
    data["locuteurs"] = fiche
    jobs.write_segments(job_id, data)
    return _fiche_locuteurs(data)


@app.get("/jobs/{job_id}/locuteurs/{code}/original")
def speaker_original(job_id: str, code: str) -> FileResponse:
    """Échantillon de la voix d'origine du locuteur (créé à la demande)."""
    from pipeline.runner import voix_des_locuteurs

    _get_status_or_404(job_id)
    data = jobs.read_segments(job_id)
    if code not in jobs.locuteurs(data):
        raise HTTPException(404, "Locuteur introuvable.")
    chemin = jobs.project_dir(job_id) / "voix" / f"reference_{code}.wav"
    if not chemin.is_file():
        fiche = jobs.locuteurs(data)
        fiche[code] = {**fiche[code], "voix": "clone"}  # force l'extraction de l'échantillon d'origine
        voix_des_locuteurs(job_id, {**data, "locuteurs": fiche})
    if not chemin.is_file():
        chemin = jobs.project_dir(job_id) / "voix" / "reference.wav"
    return FileResponse(chemin, media_type="audio/wav")


@app.get("/jobs/{job_id}/locuteurs/{code}/timbre")
def speaker_timbre_preview(job_id: str, code: str, request: Request) -> FileResponse:
    """Écoute immédiate d'un réglage de timbre : la voix choisie du locuteur (aperçu de la voix
    de la bibliothèque, ou sa voix d'origine) passée par les filtres, sans resynthèse."""
    import tempfile

    from pipeline import voice_bank

    _get_status_or_404(job_id)
    data = jobs.read_segments(job_id)
    infos = jobs.locuteurs(data).get(code)
    if infos is None:
        raise HTTPException(404, "Locuteur introuvable.")
    choix = voice_bank.trouver(infos["voix"]) if infos["voix"] != "clone" else None
    if choix is not None and voice_bank.fichier_apercu(choix["id"]).is_file():
        source = voice_bank.fichier_apercu(choix["id"])
    else:
        speaker_original(job_id, code)  # crée l'échantillon d'origine si besoin
        source = jobs.project_dir(job_id) / "voix" / f"reference_{code}.wav"
        if not source.is_file():
            source = jobs.project_dir(job_id) / "voix" / "reference.wav"
    reglages = {}
    for cle in voice_bank.TIMBRE_BORNES:
        if cle in request.query_params:
            try:
                reglages[cle] = float(request.query_params[cle])
            except ValueError:
                raise HTTPException(400, f"Réglage invalide : {cle}.") from None
    dossier = Path(tempfile.mkdtemp(prefix="voxlingo-timbre-"))
    copie = dossier / "apercu.wav"
    shutil.copy(source, copie)
    voice_bank.transformer(copie, voice_bank.nettoyer_timbre(reglages))
    return FileResponse(copie, media_type="audio/wav")


@app.get("/jobs/{job_id}/locuteurs/{code}/doublee")
def speaker_dubbed(job_id: str, code: str) -> FileResponse:
    """Première phrase doublée du locuteur, pour écouter sa voix dans la langue cible."""
    from pipeline.voice import fichier_brut

    _get_status_or_404(job_id)
    data = jobs.read_segments(job_id)
    for seg in sorted((s for s in data["segments"] if jobs.locuteur(s) == code),
                      key=lambda s: s["fin"] - s["debut"], reverse=True):
        chemin = fichier_brut(jobs.project_dir(job_id) / "voix", seg["id"])
        if chemin.is_file():
            return FileResponse(chemin, media_type="audio/wav")
    raise HTTPException(404, "Aucune voix doublée générée pour ce locuteur.")


@app.get("/jobs/{job_id}/locuteurs/{code}/avatar")
def speaker_avatar(job_id: str, code: str) -> FileResponse:
    """Image de la vidéo au milieu de la plus longue phrase du locuteur (pour sa carte)."""
    from pipeline.extract import _require

    status = _get_status_or_404(job_id)
    # Visage du personnage reconnu à l'image (lecture des scènes), s'il y en a un
    personnage = (jobs.locuteurs(jobs.read_segments(job_id)).get(code) or {}).get("personnage")
    visage = jobs.project_dir(job_id) / "visages" / f"perso_{personnage}.jpg"
    if personnage is not None and visage.is_file():
        return FileResponse(visage, media_type="image/jpeg")
    chemin = jobs.project_dir(job_id) / "voix" / f"avatar_{code}.jpg"
    if not chemin.is_file():
        data = jobs.read_segments(job_id)
        phrases = [s for s in data["segments"] if jobs.locuteur(s) == code]
        if not phrases:
            raise HTTPException(404, "Locuteur introuvable.")
        seg = max(phrases, key=lambda s: s["fin"] - s["debut"])
        chemin.parent.mkdir(exist_ok=True)
        subprocess.run(
            [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{(seg['debut'] + seg['fin']) / 2:.2f}",
             "-i", str(jobs.project_dir(job_id) / status["fichier"]), "-frames:v", "1",
             "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=256:256", "-q:v", "4", str(chemin)],
            capture_output=True,
        )
    if not chemin.is_file():
        raise HTTPException(404, "Image indisponible.")
    return FileResponse(chemin, media_type="image/jpeg")


# ───────────────────────── Bibliothèque de voix ─────────────────────────

_apercus = {"en_cours": False, "erreur": None}


@app.get("/voix")
def list_voices() -> dict:
    """Voix de la bibliothèque (XTTS intégrées et personnelles) et état de la génération des aperçus."""
    from pipeline import voice_bank

    return {"voix": voice_bank.lire_banque(), "apercus": _apercus}


@app.post("/voix/apercus")
def generate_previews() -> dict:
    """Génère en arrière-plan l'aperçu (et la mesure du registre) des voix qui n'en ont pas."""
    import threading

    from pipeline import voice_bank

    if _apercus["en_cours"]:
        return _apercus

    def travail() -> None:
        try:
            voice_bank.generer_apercus()
            _apercus["erreur"] = None
            sauvegarde.apres_changement_voix()
        except Exception as exc:  # message lisible pour l'interface
            logger.exception("Aperçus des voix")
            _apercus["erreur"] = str(exc)
        finally:
            _apercus["en_cours"] = False

    _apercus.update(en_cours=True, erreur=None)
    threading.Thread(target=travail, name="apercus-voix", daemon=True).start()
    return _apercus


@app.get("/voix/{voix_id}/apercu")
def voice_preview(voix_id: str) -> FileResponse:
    from pipeline import voice_bank

    chemin = voice_bank.fichier_apercu(voix_id)
    if not chemin.is_file():
        raise HTTPException(404, "Aperçu pas encore généré.")
    return FileResponse(chemin, media_type="audio/wav")


@app.post("/voix", status_code=201)
async def add_voice(
    echantillon: UploadFile = File(...),
    nom: str = Form(...),
    sexe: Literal["femme", "homme"] = Form(...),
    age: Literal["enfant", "jeune", "adulte", "senior"] = Form("adulte"),
) -> dict:
    """Ajoute une voix personnelle (échantillon de 6 à 15 s) à la bibliothèque."""
    import tempfile

    from pipeline import voice_bank

    suffixe = Path(echantillon.filename or "voix.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as dossier:
        temporaire = Path(dossier) / f"echantillon{suffixe}"
        temporaire.write_bytes(await echantillon.read())
        voix = voice_bank.ajouter_voix_perso(nom, sexe, age, temporaire)
    sauvegarde.apres_changement_voix()
    return voix


class ModifVoix(BaseModel):
    nom: str | None = None
    sexe: Literal["femme", "homme"] | None = None
    age: Literal["enfant", "jeune", "adulte", "senior"] | None = None


@app.put("/voix/{voix_id}")
def update_voice(voix_id: str, modif: ModifVoix) -> dict:
    """Corrige le nom, le sexe ou l'âge apparent d'une voix."""
    from pipeline import voice_bank

    banque = voice_bank.lire_banque()
    voix = next((v for v in banque if v["id"] == voix_id), None)
    if voix is None:
        raise HTTPException(404, "Voix introuvable.")
    for champ, valeur in modif.model_dump(exclude_none=True).items():
        voix[champ] = valeur.strip() if isinstance(valeur, str) and champ == "nom" else valeur
    voix["registre"] = voice_bank.registre(voix["sexe"], voix.get("f0_hz"))
    voice_bank.ecrire_banque(banque)
    sauvegarde.apres_changement_voix()
    return voix


@app.delete("/voix/{voix_id}")
def delete_voice(voix_id: str) -> dict:
    """Supprime une voix personnelle (les voix XTTS intégrées ne se suppriment pas)."""
    from pipeline import voice_bank

    try:
        voice_bank.supprimer_voix_perso(voix_id)
    except KeyError:
        raise HTTPException(400, "Seules les voix personnelles peuvent être supprimées.") from None
    sauvegarde.apres_changement_voix()
    return {"supprime": voix_id}


@app.get("/health")
def health() -> dict:
    """État du système : matériel, mode, modèles choisis et outils."""
    runtime = get_runtime()
    vram_libre = 0.0
    if runtime.device == "cuda":
        import torch

        libre, _total = torch.cuda.mem_get_info()
        vram_libre = round(libre / 1024**3, 1)
    return {
        "status": "ok",
        "device": runtime.device,
        "mode": runtime.mode,
        "vram_totale_go": runtime.vram_go,
        "vram_libre_go": vram_libre,
        "modeles": {
            "transcription": runtime.whisper_model,
            "traduction": runtime.ollama_model,
            "synchro_labiale_locale": runtime.lipsync_local,
        },
        "langues_cibles": get_settings().target_languages,
        "nom_utilisateur": get_settings().nom_utilisateur,
        "outils": tool_versions(),
    }
