"""Orchestration du pipeline : enchaîne les étapes, met à jour le statut et la progression.

Le pipeline s'arrête après la traduction (statut « a_valider ») : l'utilisateur
relit et corrige les traductions avant la génération des voix.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

import jobs
import sauvegarde
from pipeline.extract import compute_peaks, extract_audio, make_filmstrip, make_thumbnail, probe_duration
from pipeline.transcribe import transcribe
from pipeline.adjust import ajuster_segments
from pipeline.segments import est_trop_long, marquer_fin_max
from pipeline.translate import translate_segments

logger = logging.getLogger("voxlingo.runner")

# Poids relatif de chaque étape disponible dans la progression globale (en %).
WEIGHTS: dict[str, int] = {
    "extraction": 3,
    "separation": 10,
    "transcription": 9,
    "locuteurs": 3,
    "analyse": 5,
    "traduction": 20,
    "voix": 40,
    "export": 10,
    "synchro_labiale": 0,  # facultative (moteur sur GPU) : suivie par sa propre progression
}


def _progress_reporter(job_id: str, cle: str) -> Callable[[float], None]:
    """Renvoie une fonction qui convertit l'avancement d'une étape (0..1) en progression globale."""
    done_before = 0
    for key, weight in WEIGHTS.items():
        if key == cle:
            break
        done_before += weight
    weight = WEIGHTS[cle]
    last = {"etape": -1}

    def report(fraction: float) -> None:
        jobs.check_cancelled(job_id)  # point de contrôle de l'annulation
        etape = int(fraction * 100)
        if etape != last["etape"]:  # évite d'écrire status.json pour rien
            last["etape"] = etape
            jobs.update_status(
                job_id, progression=int(done_before + weight * fraction), progression_etape=etape
            )

    return report


def _run_step(job_id: str, cle: str, action: Callable[[Callable[[float], None]], None]) -> None:
    """Exécute une étape en journalisant sa durée et en mettant à jour son état."""
    libelle = next(s["libelle"] for s in jobs.STEPS if s["cle"] == cle)
    logger.info("[%s] %s : début", job_id, libelle)
    jobs.update_status(job_id, etape=cle, progression_etape=0)
    jobs.update_step(job_id, cle, etat="en_cours", duree_s=None)
    report = _progress_reporter(job_id, cle)
    start = time.perf_counter()
    try:
        action(report)
    except jobs.JobCancelled:
        jobs.update_step(job_id, cle, etat="attente", duree_s=None)  # annulée : pas un échec
        raise
    except Exception:
        jobs.update_step(job_id, cle, etat="erreur", duree_s=round(time.perf_counter() - start, 1))
        raise
    duree = round(time.perf_counter() - start, 1)
    report(1.0)
    jobs.update_step(job_id, cle, etat="termine", duree_s=duree)
    logger.info("[%s] %s : terminé en %.1f s", job_id, libelle, duree)


def _reinitialiser_etapes(job_id: str, premiere: str) -> None:
    """Avant une nouvelle exécution : l'étape `premiere` et les suivantes repassent « en attente »."""
    status = jobs.read_status(job_id)
    cles = [s["cle"] for s in status["etapes"]]
    for step in status["etapes"][cles.index(premiere):]:
        if step["etat"] != "a_venir":
            step.update(etat="attente", duree_s=None)
    jobs.write_status(job_id, status)


def _guarded(job_id: str, body: Callable[[], None], etat_final: str = "a_valider") -> None:
    """Exécute `body` en transformant annulation et erreurs en statut lisible."""
    jobs.update_status(
        job_id, etat="en_cours", erreur=None, annulation_demandee=False, debut_traitement=jobs._now()
    )
    try:
        body()
        jobs.update_status(job_id, etat=etat_final, etape=None, fin_traitement=jobs._now())
        logger.info("[%s] Terminé (%s)", job_id, etat_final)
        if etat_final == "termine":
            sauvegarde.apres_export(job_id)  # Kaggle/Colab : copie sur Hugging Face si un jeton est fourni
    except jobs.JobCancelled as exc:
        logger.info("[%s] Annulé", job_id)
        jobs.update_status(job_id, etat="annule", etape=None, erreur=str(exc), fin_traitement=jobs._now())
    except Exception as exc:  # toute erreur est remontée à l'interface
        logger.exception("[%s] Échec du pipeline", job_id)
        message = str(exc) if str(exc) else exc.__class__.__name__
        jobs.update_status(job_id, etat="erreur", erreur=message, fin_traitement=jobs._now())


# Part de l'étape « Traduction » consacrée à l'ajustement des phrases trop longues
PART_AJUSTEMENT = 0.2


def _sous_progression(report: Callable[[float], None], debut: float, fin: float) -> Callable[[float], None]:
    return lambda fraction: report(debut + (fin - debut) * fraction)


def _consignes(job_id: str):
    """Glossaire et consignes de traduction du projet (options « glossaire » et « consignes »)."""
    from pipeline.translate import consignes_du_projet

    options = (jobs.read_status(job_id) or {}).get("options") or {}
    return consignes_du_projet(options.get("glossaire", ""), options.get("consignes", ""))


def _contexte(job_id: str):
    """Fiche de la vidéo, scènes et noms des locuteurs, transmis à chaque traduction."""
    from pipeline.scene import formater_fiche
    from pipeline.translate import contexte_de_la_scene

    from pipeline.translate import genre_selon_voix

    data = jobs.read_segments(job_id)
    fiche = jobs.locuteurs(data)
    noms = {code: infos.get("nom", code) for code, infos in fiche.items()}
    genres = {code: genre_selon_voix(infos.get("f0_hz")) for code, infos in fiche.items()}
    return contexte_de_la_scene(formater_fiche(data.get("fiche")), data.get("scenes"), noms, genres)


def _analyser_scene(job_id: str, report: Callable[[float], None]) -> None:
    """Regarder et écouter avant de traduire : émotion de chaque phrase (voix, manière de dire,
    visage), description des scènes, puis fiche de la vidéo rédigée par l'IA."""
    from pipeline import emotions, scene

    status = jobs.read_status(job_id)
    dossier = jobs.project_dir(job_id)
    data = jobs.read_segments(job_id)
    vision = json.loads((dossier / "vision.json").read_text(encoding="utf-8")) if (dossier / "vision.json").is_file() else None
    voix_isolee = dossier / "voix_isolee.wav"
    source = voix_isolee if voix_isolee.is_file() else dossier / "audio.wav"
    if any("prosodie" not in s for s in data["segments"]):
        _analyser_tons(job_id, data)
    emotions.analyser(data["segments"], source, vision, jobs.locuteurs(data), on_progress=_sous_progression(report, 0, 0.4))
    jobs.write_segments(job_id, data)
    duree = status.get("duree_s") or data.get("duree_s") or 0.0
    data["scenes"] = scene.decrire_scenes(dossier / status["fichier"], vision, duree,
                                          on_progress=_sous_progression(report, 0.4, 0.8))
    jobs.write_segments(job_id, data)
    if not (data.get("fiche") or {}).get("corrigee"):
        data["fiche"] = scene.rediger_fiche(data["segments"], jobs.locuteurs(data), data["scenes"],
                                            status.get("langue_source") or data.get("langue") or "fr")
    jobs.write_segments(job_id, data)
    report(1.0)


def _do_translate(job_id: str, report: Callable[[float], None], ids: set[int] | None = None) -> None:
    with _consignes(job_id), _contexte(job_id):
        _traduire(job_id, report, ids)


def _traduire(job_id: str, report: Callable[[float], None], ids: set[int] | None = None) -> None:
    """Traduit les segments (tous, ou seulement `ids`), puis ajuste les phrases trop longues si l'option est active."""
    status = jobs.read_status(job_id)
    source, cible = status["langue_source"], status["langue_cible"]
    ajuster = status.get("options", {}).get("ajuster_trop_longs", True)
    data = jobs.read_segments(job_id)
    data["duree_s"] = status.get("duree_s") or data.get("duree_s")
    marquer_fin_max(data["segments"], data["duree_s"])
    if ids is not None:
        _retraduire_selection(job_id, data, ids, source, cible, ajuster, report)
        return

    fin_traduction = 1 - PART_AJUSTEMENT if ajuster else 1.0
    traductions, moteur = translate_segments(
        data["segments"], source, cible, on_progress=_sous_progression(report, 0, fin_traduction)
    )
    for seg, texte in zip(data["segments"], traductions, strict=True):
        seg["traduction_ia"] = texte
        seg["traduction"] = texte
        seg["a_retraduire"] = False
        seg["trop_long"] = est_trop_long(seg, texte, source, cible, jobs.strategie(status))
    data["langue_cible"] = cible
    data["moteur_traduction"] = moteur
    jobs.write_segments(job_id, data)

    bilan = None
    if ajuster and any(seg["trop_long"] for seg in data["segments"]):
        bilan = ajuster_segments(data, source, cible, on_progress=_sous_progression(report, fin_traduction, 1),
                                 strategie=jobs.strategie(status))
        jobs.write_segments(job_id, data)
    jobs.update_status(job_id, moteur_traduction=moteur, traduction_validee=False, dernier_ajustement=bilan)


def _retraduire_selection(job_id: str, data: dict, ids: set[int], source: str, cible: str,
                           ajuster: bool, report: Callable[[float], None]) -> None:
    """Retraduit seulement les phrases dont le texte d'origine a été corrigé ; les autres sont conservées."""
    from pipeline.translate import traduire_selection

    status = jobs.read_status(job_id)
    nouvelles = traduire_selection(data["segments"], ids, source, cible,
                                   on_progress=_sous_progression(report, 0, 0.8 if ajuster else 1.0))
    for seg in data["segments"]:
        if seg["id"] in nouvelles:
            seg["traduction_ia"] = seg["traduction"] = nouvelles[seg["id"]]
            seg["a_retraduire"] = False
            seg["trop_long"] = est_trop_long(seg, seg["traduction"], source, cible, jobs.strategie(status))
    jobs.write_segments(job_id, data)
    if ajuster and any(seg["trop_long"] for seg in data["segments"] if seg["id"] in nouvelles):
        bilan = ajuster_segments(data, source, cible, ids=set(nouvelles), strategie=jobs.strategie(status),
                                 on_progress=_sous_progression(report, 0.8, 1.0))
        jobs.write_segments(job_id, data)
        jobs.update_status(job_id, dernier_ajustement=bilan)
    jobs.update_status(job_id, traduction_validee=False)


def _do_adjust(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    with _consignes(job_id), _contexte(job_id):
        _ajuster(job_id, ids, report)


def _ajuster(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    """Ajuste les phrases trop longues après les modifications de l'utilisateur."""
    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    data["duree_s"] = status.get("duree_s") or data.get("duree_s")
    bilan = ajuster_segments(data, status["langue_source"], status["langue_cible"], ids=ids, on_progress=report,
                             strategie=jobs.strategie(status))
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, traduction_validee=False, dernier_ajustement=bilan)


# File d'attente : un seul nouveau projet traité à la fois (plusieurs langues lancées d'un coup,
# plusieurs vidéos…) ; les suivants restent « en attente » sans saturer la mémoire.
_FILE_ATTENTE = threading.Semaphore(1)


def run_pipeline(job_id: str) -> None:
    """Point d'entrée du thread de traitement d'une nouvelle tâche (attend son tour dans la file)."""
    with _FILE_ATTENTE:
        status = jobs.read_status(job_id)
        if status is not None and status.get("annulation_demandee"):
            # Annulé pendant qu'il attendait son tour
            jobs.update_status(job_id, etat="annule", erreur="Traitement annulé par l'utilisateur.",
                               annulation_demandee=False, fin_traitement=jobs._now())
            return
        _traiter(job_id)


def _traiter(job_id: str) -> None:
    status = jobs.read_status(job_id)
    if status is None:
        logger.error("Tâche %s introuvable", job_id)
        return

    folder = jobs.project_dir(job_id)
    video = folder / status["fichier"]
    audio = folder / "audio.wav"
    jobs.update_status(job_id, progression=0)

    def do_extract(report: Callable[[float], None]) -> None:
        nonlocal video
        decoupe = (status.get("options") or {}).get("decoupe")
        if decoupe and not jobs.read_status(job_id).get("decoupe_appliquee"):
            # Option « Raccourcir la vidéo » : la suite du traitement ne voit que le passage gardé
            from pipeline.format import couper_meme_format

            # Même conteneur, codecs, résolution et cadence que la vidéo fournie
            coupee = couper_meme_format(video, folder / f"input_coupee{video.suffix}", decoupe["debut_s"], decoupe["fin_s"])
            video.unlink()
            video = coupee.replace(folder / f"input{video.suffix}")
            jobs.update_status(job_id, fichier=video.name, decoupe_appliquee=True)
            logger.info("[%s] Vidéo raccourcie : %.1f s → %.1f s", job_id, decoupe["debut_s"], decoupe["fin_s"])
            report(0.2)
        duree = probe_duration(video)
        jobs.update_status(job_id, duree_s=round(duree, 2))
        make_thumbnail(video, folder / "thumbnail.jpg", duree)
        make_filmstrip(video, folder / "filmstrip.jpg", duree)
        report(0.3)
        extract_audio(video, audio)
        (folder / "waveform.json").write_text(json.dumps(compute_peaks(audio)), encoding="utf-8")

    def do_separate(report: Callable[[float], None]) -> None:
        from pipeline.separate import separer

        separer(video, folder, on_progress=report)

    def do_transcribe(report: Callable[[float], None]) -> None:
        voix_isolee = folder / "voix_isolee.wav"
        result = transcribe(voix_isolee if voix_isolee.is_file() else audio, on_progress=report)
        jobs.write_segments(job_id, result.to_dict())
        jobs.update_status(job_id, langue_source=result.langue, probabilite_langue=result.probabilite)

    def do_speakers(report: Callable[[float], None]) -> None:
        _identifier_locuteurs(job_id, report)

    def body() -> None:
        _run_step(job_id, "extraction", do_extract)
        if (status.get("options") or {}).get("conserver_fond", True):
            _run_step(job_id, "separation", do_separate)
        else:
            jobs.update_step(job_id, "separation", etat="ignore")
        _run_step(job_id, "transcription", do_transcribe)
        _run_step(job_id, "locuteurs", do_speakers)
        _run_step(job_id, "analyse", lambda report: _analyser_scene(job_id, report))
        _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report))

    _guarded(job_id, body)


def _mesurer_locuteurs(job_id: str, data: dict) -> None:
    """Mesure la hauteur de voix (f0) de chaque locuteur sur ses phrases les plus longues."""
    import numpy as np

    from pipeline.voice_bank import mesurer_f0

    audio = jobs.project_dir(job_id) / "audio.wav"
    for code, infos in jobs.locuteurs(data).items():
        phrases = sorted((s for s in data["segments"] if jobs.locuteur(s) == code),
                         key=lambda s: s["fin"] - s["debut"], reverse=True)[:3]
        mesures = [f for s in phrases if (f := mesurer_f0(audio, s["debut"], s["fin"] - s["debut"])) is not None]
        infos["f0_hz"] = round(float(np.median(mesures)), 1) if mesures else None
        data["locuteurs"][code] = infos


def _analyser_tons(job_id: str, data: dict, report: Callable[[float], None] | None = None) -> None:
    """Ton de chaque phrase (énergique, calme…) et écart de volume à la moyenne de son locuteur."""
    from pipeline.prosody import analyser, marquer_ecarts

    analyser(data["segments"], jobs.project_dir(job_id) / "audio.wav", on_progress=report)
    marquer_ecarts(data["segments"])


def _identifier_locuteurs(job_id: str, report: Callable[[float], None]) -> None:
    """Identifie qui parle dans chaque phrase, mesure chaque voix et, si l'option est active,
    attribue automatiquement une voix de la bibliothèque à chaque locuteur."""
    from pipeline import voice_bank
    from pipeline.diarize import identifier_locuteurs

    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    dossier = jobs.project_dir(job_id)
    voix_isolee = dossier / "voix_isolee.wav"  # sans musique ni bruitages : hauteurs et empreintes plus nettes
    # Avant la traduction, une phrase dite par plusieurs personnes est découpée en répliques ;
    # après, on garde les phrases (et leurs traductions) telles quelles.
    deja_traduit = any(s.get("traduction") for s in data["segments"])
    # Lecture des scènes : qui est à l'image et quelle bouche bouge (facultative : la voix suffit sans)
    vision = None
    try:
        from pipeline.vision import analyser

        vision = analyser(dossier / status["fichier"], dossier, status.get("duree_s"),
                          on_progress=_sous_progression(report, 0, 0.4))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Lecture des scènes impossible (%s) : identification par la voix seule", job_id, exc)
    data["locuteurs"] = identifier_locuteurs(
        data["segments"], voix_isolee if voix_isolee.is_file() else dossier / "audio.wav",
        on_progress=_sous_progression(report, 0.4, 0.85), decouper=not deja_traduit, vision=vision,
    )
    _mesurer_locuteurs(job_id, data)
    _analyser_tons(job_id, data, _sous_progression(report, 0.85, 1.0))
    if (status.get("options") or {}).get("voix_bibliotheque", False):
        for code, voix_id in voice_bank.attribuer(data["locuteurs"]).items():
            data["locuteurs"][code]["voix"] = voix_id
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, nb_locuteurs=len(data["locuteurs"]))


def run_analysis(job_id: str) -> None:
    """(Re)lance l'analyse de la scène d'un projet déjà transcrit (bouton « Analyser la scène »)."""
    etat_final = "termine" if (jobs.read_status(job_id) or {}).get("fichier_sortie") else "a_valider"
    _guarded(job_id, lambda: _run_step(job_id, "analyse", lambda report: _analyser_scene(job_id, report)),
             etat_final=etat_final)


def run_speakers(job_id: str) -> None:
    """Identifie les locuteurs d'un projet déjà transcrit (projets créés avant la phase 6)."""
    etat_final = "termine" if (jobs.read_status(job_id) or {}).get("fichier_sortie") else "a_valider"
    _guarded(job_id, lambda: _run_step(job_id, "locuteurs", lambda report: _identifier_locuteurs(job_id, report)),
             etat_final=etat_final)


def run_adjustment(job_id: str, ids: set[int] | None = None) -> None:
    """Ajuste les phrases trop longues (bouton « Ajuster automatiquement » de l'éditeur)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _reinitialiser_etapes(job_id, "traduction")
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_adjust(job_id, ids, report)))


def run_translation(job_id: str, ids: set[int] | None = None) -> None:
    """Relance la traduction : toutes les phrases (« Traduction assistée par IA ») ou seulement `ids`
    (« Retraduire les phrases corrigées »)."""
    jobs.update_status(job_id, progression=WEIGHTS["extraction"] + WEIGHTS["transcription"])
    _reinitialiser_etapes(job_id, "traduction")
    _guarded(job_id, lambda: _run_step(job_id, "traduction", lambda report: _do_translate(job_id, report, ids)))


# ───────────────────────── Phase 4 : voix et export ─────────────────────────

REFERENCE_MIN_LOCUTEUR_S = 3.0  # en dessous, la voix d'un locuteur est trop peu entendue pour la cloner


def _timbre_du_locuteur(infos: dict) -> dict:
    from pipeline.voice_bank import nettoyer_timbre

    return nettoyer_timbre(infos.get("timbre"))


def voix_des_locuteurs(job_id: str, data: dict) -> dict[str, str]:
    """Voix de chaque locuteur : sa voix d'origine clonée, ou celle choisie dans la bibliothèque."""
    from pipeline import voice_bank
    from pipeline.voice import extraire_reference

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    dossier = folder / "voix"
    dossier.mkdir(exist_ok=True)
    voix_isolee = folder / "voix_isolee.wav"
    source = voix_isolee if voix_isolee.is_file() else folder / status["fichier"]

    locuteurs = jobs.locuteurs(data)
    resultat: dict[str, str] = {}
    prises = {infos["voix"] for infos in locuteurs.values() if infos["voix"] != "clone"}
    for code, infos in locuteurs.items():
        choix = voice_bank.trouver(infos["voix"]) if infos["voix"] != "clone" else None
        phrases = [s for s in data["segments"] if jobs.locuteur(s) == code]
        if choix is None and sum(s["fin"] - s["debut"] for s in phrases) < REFERENCE_MIN_LOCUTEUR_S:
            # Trop peu de parole pour cloner sa voix : une voix de la bibliothèque, différente de celles
            # des autres locuteurs et proche de sa hauteur de voix (et non une voix commune à tous)
            choix = voice_bank.trouver(voice_bank.attribuer({code: infos}, exclues=prises)[code])
            if choix is not None:
                logger.info("[%s] %s parle trop peu pour être cloné : voix %s", job_id, code, choix["nom"])
        if choix is not None:
            prises.add(choix["id"])
            resultat[code] = voice_bank.spec(choix)
        else:
            reference = dossier / f"reference_{code}.wav"
            if not reference.is_file():
                autres = [s for s in data["segments"] if jobs.locuteur(s) != code]
                extraire_reference(source, phrases, reference, autres)
            resultat[code] = f"ref:{reference}"
        # Réglages de timbre du locuteur (page Locuteurs), appliqués après la synthèse
        timbre = voice_bank.nettoyer_timbre(infos.get("timbre"))
        if timbre:
            resultat[code] = "|".join([resultat[code], *(f"{k}={v}" for k, v in sorted(timbre.items()))])
    return resultat


def _do_voices(job_id: str, report: Callable[[float], None]) -> None:
    """Génère chaque phrase traduite avec la voix de son locuteur (seules les phrases modifiées sont refaites)."""
    from pipeline.voice import generer_voix

    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    if any("ton" not in s for s in data["segments"]):
        _analyser_tons(job_id, data)
        jobs.write_segments(job_id, data)
    par_locuteur = voix_des_locuteurs(job_id, data)
    if (status.get("options") or {}).get("intonation_originale", True):
        _ajouter_intonation(job_id, status, data, par_locuteur)
    report(0.02)
    generees = generer_voix(
        data["segments"], status["langue_cible"],
        {s["id"]: par_locuteur[jobs.locuteur(s)] for s in data["segments"]},
        jobs.project_dir(job_id) / "voix",
        on_progress=_sous_progression(report, 0.02, 1.0),
    )
    # Voix réellement utilisée pour chaque locuteur : l'interface signale un changement non généré
    data = jobs.read_segments(job_id)
    fiche = jobs.locuteurs(data)
    for infos in fiche.values():
        infos["voix_generee"] = infos["voix"]
        infos["timbre_genere"] = _timbre_du_locuteur(infos)
    data["locuteurs"] = fiche
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, voix_generees=generees)


def _ajouter_intonation(job_id: str, status: dict, data: dict, par_locuteur: dict[str, str]) -> None:
    """Voix clonées : chaque phrase reçoit l'extrait de sa phrase d'origine, qui guide l'intonation.

    Seulement pour une voix réellement clonée (« ref: ») : un locuteur qui parle trop peu reçoit une
    voix de la bibliothèque (« xtts: ») même si « clone » est demandé, et garde alors son propre style."""
    from pipeline.voice import extraire_intonation

    folder = jobs.project_dir(job_id)
    voix_isolee = folder / "voix_isolee.wav"
    source = voix_isolee if voix_isolee.is_file() else folder / status["fichier"]
    fiche = jobs.locuteurs(data)
    n = 0
    for seg in data["segments"]:
        code = jobs.locuteur(seg)
        if (fiche.get(code) or {}).get("voix", "clone") != "clone" or not par_locuteur.get(code, "").startswith("ref:"):
            continue  # voix de la bibliothèque : son propre style
        extrait = extraire_intonation(source, seg, folder / "voix" / f"intonation_{seg['id']:05d}.wav")
        if extrait is not None:
            seg["intonation"] = str(extrait)
            n += 1
    logger.info("[%s] Intonation d'origine pour %d phrase(s)", job_id, n)


def _do_export(job_id: str, report: Callable[[float], None]) -> None:
    """Cale chaque voix sur son temps, construit la piste doublée et exporte vidéo et sous-titres."""
    from pipeline.align import caler
    from pipeline.assemble import construire_piste, ecrire_srt, mixer_fond
    from pipeline.format import apercu_navigateur, exporter_meme_format
    from pipeline.prosody import gain_volume
    from pipeline.voice import fichier_brut

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    dossier = folder / "voix"
    data = jobs.read_segments(job_id)
    duree = status.get("duree_s") or data.get("duree_s")
    marquer_fin_max(data["segments"], duree)

    # Fidélité à la voix d'origine : timbre (égalisation par locuteur cloné) et volume au décibel près
    from pipeline.niveau import lire_extrait, niveau_actif_db
    from pipeline.timbre_eq import correction_locuteur, filtre_ffmpeg

    voix_isolee = folder / "voix_isolee.wav"
    egaliseurs: dict[str, str | None] = {}
    if voix_isolee.is_file():
        for code in jobs.locuteurs(data):
            if not (dossier / f"reference_{code}.wav").is_file():
                continue  # voix de la bibliothèque : son timbre est choisi, pas recopié
            phrases = [s for s in data["segments"] if jobs.locuteur(s) == code]
            gains = correction_locuteur(voix_isolee, phrases, [fichier_brut(dossier, s["id"]) for s in phrases])
            egaliseurs[code] = filtre_ffmpeg(gains)
            if gains is not None:
                logger.info("[%s] Timbre de %s : correction %s dB", job_id, code,
                            " ".join(f"{g:+.0f}" for g in gains))
    report(0.05)

    placements = []
    for n, seg in enumerate(data["segments"]):
        cale = dossier / f"cale_{seg['id']:05d}.wav"
        niveau = niveau_actif_db(*lire_extrait(voix_isolee, seg["debut"], seg["fin"])) if voix_isolee.is_file() else None
        resultat = caler(fichier_brut(dossier, seg["id"]), cale, seg, gain_volume(seg),
                         egaliseurs.get(jobs.locuteur(seg)), niveau)
        seg["niveau_origine_db"] = niveau
        seg["gain_db"] = resultat["gain_db"]
        seg["facteur_voix"] = resultat["facteur"]
        seg["duree_doublage"] = resultat["duree"]
        seg["trop_long"] = resultat["trop_long"]  # mesure réelle, remplace l'estimation
        placements.append((seg["debut"], cale))
        report(0.05 + 0.45 * (n + 1) / len(data["segments"]))
    jobs.write_segments(job_id, data)

    construire_piste(placements, duree, folder / "voix_doublee.wav")
    report(0.6)
    options = status.get("options") or {}
    if options.get("conserver_fond", True):
        from pipeline.separate import separer

        _, fond = separer(folder / status["fichier"], folder)  # projets créés avant la phase 5
        mixer_fond(folder / "voix_doublee.wav", fond, int(options.get("volume_fond", 100)), folder / "doublage.wav")
    else:
        (folder / "voix_doublee.wav").replace(folder / "doublage.wav")
    report(0.7)
    # Vidéo doublée au format exact de la vidéo fournie (conteneur, codecs, résolution, cadence, son)
    video = folder / status["fichier"]
    sortie = folder / f"doublage{video.suffix.lower()}"
    exporter_meme_format(video, folder / "doublage.wav", sortie)
    apercu_navigateur(sortie, folder / "apercu.mp4")  # lecture dans le site, si ce format ne s'y lit pas
    if status.get("options", {}).get("sous_titres", True):
        ecrire_srt(data["segments"], folder / "sous-titres.srt")
    jobs.update_status(
        job_id,
        fichier_sortie=sortie.name,
        segments_trop_longs=sum(1 for s in data["segments"] if s["trop_long"]),
    )


def run_voices(job_id: str) -> None:
    """Génère les voix puis exporte la vidéo doublée (après « Valider et générer les voix »)."""
    jobs.update_status(
        job_id,
        progression=WEIGHTS["extraction"] + WEIGHTS["transcription"] + WEIGHTS["traduction"],
        fichier_sortie=None,
    )
    _reinitialiser_etapes(job_id, "voix")

    def body() -> None:
        _run_step(job_id, "voix", lambda report: _do_voices(job_id, report))
        _run_step(job_id, "export", lambda report: _do_export(job_id, report))
        jobs.update_status(job_id, synchro_importee=False)  # la piste a changé : l'ancienne synchro n'est plus valable
        options = (jobs.read_status(job_id) or {}).get("options") or {}
        if not options.get("synchro_labiale"):
            jobs.update_step(job_id, "synchro_labiale", etat="ignore")
        else:
            _synchro_labiale(job_id)

    _guarded(job_id, body, etat_final="termine")


def _synchro_labiale(job_id: str) -> None:
    """Synchronisation labiale par le moteur (MuseTalk, GPU) ; sinon, ou si elle échoue, elle reste à
    faire avec le carnet Kaggle manuel (écran Résultat). Un échec ne fait jamais échouer le doublage."""
    from pipeline.lipsync import musetalk_disponible

    if not musetalk_disponible():
        jobs.update_step(job_id, "synchro_labiale", etat="kaggle")
        return
    try:
        _run_step(job_id, "synchro_labiale", lambda report: _do_lipsync(job_id, report))
    except jobs.JobCancelled:
        raise
    except Exception as exc:
        logger.exception("[%s] Synchronisation labiale", job_id)
        jobs.update_step(job_id, "synchro_labiale", etat="kaggle")
        jobs.update_status(job_id, synchro_erreur=str(exc) or exc.__class__.__name__)


def _do_lipsync(job_id: str, report: Callable[[float], None]) -> None:
    from pipeline.extract import probe_duration
    from pipeline.format import apercu_navigateur, conformer
    from pipeline.lipsync import synchroniser

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    video = folder / status["fichier"]
    data = jobs.read_segments(job_id)
    resultat = synchroniser(video, folder / "doublage.wav", data["segments"], probe_duration(video), folder, report)
    for ancienne in folder.glob("doublage_synchro.*"):
        ancienne.unlink()
    cible = conformer(resultat, video, folder / f"doublage_synchro{video.suffix.lower()}")
    resultat.unlink(missing_ok=True)
    apercu_navigateur(cible, folder / "apercu_synchro.mp4")
    jobs.update_status(job_id, synchro_importee=True, synchro_moteur=True, synchro_erreur=None)


def run_lipsync(job_id: str) -> None:
    """Synchronisation labiale demandée après coup depuis l'écran Résultat (moteur sur GPU)."""
    jobs.update_status(job_id, synchro_erreur=None)
    _guarded(job_id, lambda: _synchro_labiale(job_id), etat_final="termine")
