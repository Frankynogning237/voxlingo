"""Tests de la synchronisation labiale par le moteur (MuseTalk simulé, vrai FFmpeg)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import lipsync

# Faux MuseTalk : « synchronise » en recopiant chaque vidéo source ; échoue sur les tâches demandées
FAUX_MUSETALK = '''
import argparse, os, re, shutil, sys, time
if os.environ.get("FIGE"):
    time.sleep(60)  # MuseTalk figé : plus rien ne sort
p = argparse.ArgumentParser()
p.add_argument("--inference_config"); p.add_argument("--result_dir")
p.add_argument("--unet_model_path"); p.add_argument("--unet_config"); p.add_argument("--version")
p.add_argument("--use_float16", action="store_true"); p.add_argument("--batch_size")
a = p.parse_args()
texte = open(a.inference_config, encoding="utf-8").read()
os.makedirs(os.path.join(a.result_dir, "v15"), exist_ok=True)
echecs = os.environ.get("ECHECS", "").split(",")
print("Loaded inference config:", a.inference_config, flush=True)
for nom, video, nom_sortie in re.findall(r'(tache_\\d+):\\n  video_path: "([^"]+)"\\n  audio_path: "[^"]+"\\n'
                                         r'  result_name: "([^"]+)"', texte):
    print(nom, flush=True)
    if os.environ.get("LENT"):  # barre tqdm à mi-chemin, sans saut de ligne, puis un long calcul muet
        sys.stderr.write(" 50%|#####     | 5/10 [00:01<00:01]\\r"); sys.stderr.flush(); time.sleep(float(os.environ["LENT"]))
    if nom not in echecs:
        shutil.copy(video, os.path.join(a.result_dir, "v15", nom_sortie))
'''

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg absent")


def test_fenetres_fusionne_et_decoupe():
    segments = [
        {"debut": 1.0, "fin": 2.0, "traduction": "Bonjour"},
        {"debut": 2.3, "fin": 3.0, "fin_max": 3.5, "traduction": "ça va ?"},  # fusionnée (écart < 0,6 s)
        {"debut": 10.0, "fin": 11.0, "traduction": ""},  # pas de voix doublée : ignorée
        {"debut": 20.0, "fin": 52.0, "traduction": "long monologue"},  # découpé en morceaux ≤ 15 s
    ]
    morceaux = lipsync.fenetres(segments, duree=60.0)
    assert morceaux[0] == (0.8, 3.7)
    longs = morceaux[1:]
    assert len(longs) == 3 and longs[0][0] == 19.8 and longs[-1][1] == 52.2
    assert all(f - d <= lipsync.MORCEAU_MAX_S for d, f in longs)
    assert all(longs[i][1] == longs[i + 1][0] for i in range(len(longs) - 1))


def test_plan_de_montage_couvre_toute_la_duree():
    plan = lipsync.plan_de_montage([(1.0, 2.0), (2.0, 3.0), (5.0, 6.0)], duree=8.0)
    assert plan == [(0.0, 1.0, None), (1.0, 2.0, 0), (2.0, 3.0, 1), (3.0, 5.0, None), (5.0, 6.0, 2), (6.0, 8.0, None)]
    assert lipsync.plan_de_montage([(0.0, 8.0)], duree=8.0) == [(0.0, 8.0, 0)]


def test_images_sans_derive():
    bornes = [0.0, 0.84, 1.96, 3.0, 7.48, 10.0]
    total = sum(lipsync.images(bornes[i], bornes[i + 1]) for i in range(len(bornes) - 1))
    assert total == 10 * lipsync.CADENCE


def test_disponible_seulement_une_fois_installe(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("")
    reglages = SimpleNamespace(musetalk_python=str(python), musetalk_dir=str(tmp_path))
    monkeypatch.setattr(lipsync, "get_settings", lambda: reglages)
    assert not lipsync.musetalk_disponible()  # poids pas encore téléchargés
    (tmp_path / ".pret").write_text("ok")
    assert lipsync.musetalk_disponible()
    monkeypatch.setattr(lipsync, "get_settings", lambda: SimpleNamespace(musetalk_python=None, musetalk_dir=None))
    assert not lipsync.musetalk_disponible()


@pytest.fixture
def projet(tmp_path, monkeypatch):
    """Vidéo de 6 s (30 images/s, taille impaire), piste doublée, faux MuseTalk."""
    musetalk = tmp_path / "musetalk"
    (musetalk / "scripts").mkdir(parents=True)
    (musetalk / "scripts" / "__init__.py").write_text("")
    (musetalk / "scripts" / "inference.py").write_text(FAUX_MUSETALK, encoding="utf-8")
    monkeypatch.setattr(lipsync, "get_settings",
                        lambda: SimpleNamespace(musetalk_python=sys.executable, musetalk_dir=str(musetalk)))
    video, piste = tmp_path / "video.mp4", tmp_path / "doublage.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=321x241:rate=30:duration=6",
                    "-pix_fmt", "yuv444p", str(video)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                    str(piste)], check=True)
    segments = [{"debut": 0.5, "fin": 2.0, "traduction": "Hello"}, {"debut": 3.1, "fin": 4.2, "traduction": "Bye"}]
    return tmp_path, video, piste, segments


def _sonder(fichier: Path) -> dict:
    sortie = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                             "stream=codec_type,nb_read_frames,width,height:format=duration", "-of", "json",
                             str(fichier)], capture_output=True, text=True, check=True).stdout
    return json.loads(sortie)


@ffmpeg
def test_synchroniser_recolle_toute_la_video(projet, monkeypatch):
    dossier, video, piste, segments = projet
    monkeypatch.setenv("ECHECS", "tache_0001")  # un passage sans visage : il garde l'image d'origine
    progression = []
    sortie = lipsync.synchroniser(video, piste, segments, 6.0, dossier, progression.append)
    infos = _sonder(sortie)
    image = next(s for s in infos["streams"] if s["codec_type"] == "video")
    assert int(image["nb_read_frames"]) == 6 * lipsync.CADENCE  # aucune image perdue ni ajoutée
    assert (image["width"], image["height"]) == (320, 240)
    assert any(s["codec_type"] == "audio" for s in infos["streams"])
    assert abs(float(infos["format"]["duration"]) - 6.0) < 0.1
    assert progression[-1] == pytest.approx(0.95)
    assert not (dossier / "synchro").exists()  # fichiers de travail supprimés


@ffmpeg
def test_synchroniser_sans_aucun_resultat(projet, monkeypatch):
    dossier, video, piste, segments = projet
    monkeypatch.setenv("ECHECS", "tache_0000,tache_0001")
    with pytest.raises(lipsync.LipsyncError, match="aucun passage"):
        lipsync.synchroniser(video, piste, segments, 6.0, dossier)


def test_echec_de_la_synchro_ne_fait_pas_echouer_le_doublage(monkeypatch):
    import jobs
    from pipeline import runner

    etapes, statuts = [], []
    monkeypatch.setattr(jobs, "update_step", lambda job_id, cle, **c: etapes.append((cle, c.get("etat"))))
    monkeypatch.setattr(jobs, "update_status", lambda job_id, **c: statuts.append(c))

    monkeypatch.setattr(lipsync, "musetalk_disponible", lambda: False)
    runner._synchro_labiale("abc")
    assert etapes == [("synchro_labiale", "kaggle")]  # sans GPU : carnet Kaggle manuel

    etapes.clear()
    monkeypatch.setattr(lipsync, "musetalk_disponible", lambda: True)

    def echoue(job_id, cle, action):
        raise lipsync.LipsyncError("visages introuvables")

    monkeypatch.setattr(runner, "_run_step", echoue)
    runner._synchro_labiale("abc")  # ne lève rien
    assert etapes == [("synchro_labiale", "kaggle")]
    assert statuts[-1] == {"synchro_erreur": "visages introuvables"}


@ffmpeg
def test_progression_avance_pendant_un_passage(projet, monkeypatch):
    """Pendant un long passage muet, la progression suit la barre de MuseTalk au lieu de rester à 10 %."""
    dossier, video, piste, segments = projet
    monkeypatch.setenv("LENT", "1.5")
    monkeypatch.setattr(lipsync, "SONDAGE_S", 0.05)
    progression = []
    lipsync.synchroniser(video, piste, segments, 6.0, dossier, progression.append)
    assert any(0.1 < p < 0.85 for p in progression)
    assert progression == sorted(progression)  # ne recule jamais


@ffmpeg
def test_musetalk_fige_est_arrete(projet, monkeypatch):
    dossier, video, piste, segments = projet
    monkeypatch.setenv("FIGE", "1")
    monkeypatch.setattr(lipsync, "SILENCE_MAX_S", 1)
    monkeypatch.setattr(lipsync, "SONDAGE_S", 0.05)
    with pytest.raises(lipsync.LipsyncError, match="ne répond plus"):
        lipsync.synchroniser(video, piste, segments, 6.0, dossier)


def test_progression_donne_signe_de_vie(monkeypatch):
    """Même pourcentage répété : status.json est quand même réécrit après SIGNE_DE_VIE_S."""
    from pipeline import runner

    ecritures = []
    monkeypatch.setattr(runner.jobs, "check_cancelled", lambda job_id: None)
    monkeypatch.setattr(runner.jobs, "update_status", lambda job_id, **c: ecritures.append(c))
    report = runner._progress_reporter("p1", "synchro_labiale")
    report(0.1)
    report(0.1)
    assert len(ecritures) == 1
    monkeypatch.setattr(runner, "SIGNE_DE_VIE_S", 0)
    report(0.1)
    assert len(ecritures) == 2


def test_redemarrage_pendant_la_synchro_garde_le_doublage(tmp_path, monkeypatch):
    """Moteur coupé pendant la synchro : le projet reste terminé (vidéo doublée), pas renvoyé à l'édition."""
    import jobs
    import main
    from config import get_settings

    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    for job_id, exporte in (("synchro", True), ("voix", False)):
        jobs.project_dir(job_id).mkdir(parents=True)
        jobs.create_job(job_id, "video.mp4", "input.mp4", "de", {"synchro_labiale": True})
        status = jobs.read_status(job_id)
        for etape in status["etapes"]:
            if etape["cle"] in ("extraction", "voix") or (exporte and etape["cle"] == "export"):
                etape["etat"] = "termine"
        jobs.write_status(job_id, status)
        if exporte:
            (jobs.project_dir(job_id) / "doublage.mp4").write_bytes(b"video")
        jobs.update_status(job_id, etat="en_cours", etape="synchro_labiale" if exporte else "export",
                           fichier_sortie="doublage.mp4" if exporte else None)
    main.startup()
    synchro, voix = jobs.read_status("synchro"), jobs.read_status("voix")
    assert synchro["etat"] == "termine" and "interrompue" in synchro["synchro_erreur"]
    assert next(s for s in synchro["etapes"] if s["cle"] == "synchro_labiale")["etat"] == "kaggle"
    assert voix["etat"] == "erreur"  # coupé avant l'export : rien d'utilisable


def test_repartir_equilibre_les_durees():
    assert lipsync.repartir([10.0, 2.0, 7.0, 4.0], 2) == [[0, 1], [2, 3]]
    assert lipsync.repartir([3.0], 2) == [[0]]  # pas de processus sans passage
    assert lipsync.repartir([1.0, 1.0], 1) == [[0, 1]]


@ffmpeg
def test_synchroniser_sur_deux_cartes(projet, monkeypatch):
    """Deux cartes graphiques : deux MuseTalk en parallèle, chacun ses passages, tous recollés."""
    dossier, video, piste, segments = projet
    monkeypatch.setattr(lipsync, "_cartes_graphiques", lambda: 2)
    monkeypatch.setattr(lipsync, "SONDAGE_S", 0.05)
    lancements = []
    popen = subprocess.Popen

    def compter(commande, **options):
        if "--inference_config" in commande:  # FFmpeg passe aussi par Popen
            configuration = Path(commande[commande.index("--inference_config") + 1]).read_text(encoding="utf-8")
            lancements.append((configuration, options["env"].get("CUDA_VISIBLE_DEVICES")))
        return popen(commande, **options)

    monkeypatch.setattr(lipsync.subprocess, "Popen", compter)
    progression = []
    sortie = lipsync.synchroniser(video, piste, segments, 6.0, dossier, progression.append)
    assert sorted(carte for _, carte in lancements) == ["0", "1"]
    assert all(texte.count("tache_") == 1 for texte, _ in lancements)
    image = next(s for s in _sonder(sortie)["streams"] if s["codec_type"] == "video")
    assert int(image["nb_read_frames"]) == 6 * lipsync.CADENCE
    assert progression == sorted(progression)
