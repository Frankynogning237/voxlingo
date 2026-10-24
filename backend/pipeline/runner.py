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
from pipeline.adjust import a_ajuster, ajuster_segments
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
    """Traduit les segments (tous, ou seulement `ids`), puis ajuste la durée des phrases si l'option est active
    (trop longues, longues à caler sur les pauses d'origine, trop courtes)."""
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
        seg.pop("allongement_essaye", None)
        seg["trop_long"] = est_trop_long(seg, texte, source, cible, jobs.strategie(status))
    data["langue_cible"] = cible
    data["moteur_traduction"] = moteur
    jobs.write_segments(job_id, data)

    bilan = None
    if ajuster and a_ajuster(data["segments"], source, cible, jobs.strategie(status)):
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
            seg.pop("allongement_essaye", None)
            seg["trop_long"] = est_trop_long(seg, seg["traduction"], source, cible, jobs.strategie(status))
    jobs.write_segments(job_id, data)
    if ajuster and a_ajuster(data["segments"], source, cible, jobs.strategie(status), ids=set(nouvelles)):
        bilan = ajuster_segments(data, source, cible, ids=set(nouvelles), strategie=jobs.strategie(status),
                                 on_progress=_sous_progression(report, 0.8, 1.0))
        jobs.write_segments(job_id, data)
        jobs.update_status(job_id, dernier_ajustement=bilan)
    jobs.update_status(job_id, traduction_validee=False)


def _do_adjust(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    with _consignes(job_id), _contexte(job_id):
        _ajuster(job_id, ids, report)


def _ajuster(job_id: str, ids: set[int] | None, report: Callable[[float], None]) -> None:
    """Ajuste la durée des phrases (demande de l'utilisateur depuis l'éditeur)."""
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
        from pipeline.transcribe import termes_du_glossaire

        voix_isolee = folder / "voix_isolee.wav"
        termes = termes_du_glossaire((status.get("options") or {}).get("glossaire", ""))
        result = transcribe(voix_isolee if voix_isolee.is_file() else audio, on_progress=report, termes=termes)
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
    from contextlib import ExitStack

    from pipeline import voice_bank
    from pipeline.voice import extraire_reference, fichier_reference

    status = jobs.read_status(job_id)
    folder = jobs.project_dir(job_id)
    dossier = folder / "voix"
    dossier.mkdir(exist_ok=True)
    voix_isolee = folder / "voix_isolee.wav"
    source = voix_isolee if voix_isolee.is_file() else folder / status["fichier"]

    locuteurs = jobs.locuteurs(data)
    resultat: dict[str, str] = {}
    prises = {infos["voix"] for infos in locuteurs.values() if infos["voix"] != "clone"}
    # Modèle d'empreinte vocale chargé seulement si un échantillon est à composer, puis libéré
    modele: dict[str, object] = {}
    with ExitStack() as pile:
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
                reference = fichier_reference(dossier, code)
                if not reference.is_file():
                    autres = [s for s in data["segments"] if jobs.locuteur(s) != code]
                    extraire_reference(source, phrases, reference, autres,
                                       _scores_echantillon(job_id, code, source, phrases, pile, modele))
                resultat[code] = f"ref:{reference}"
            # Réglages de timbre du locuteur (page Locuteurs), appliqués après la synthèse
            timbre = voice_bank.nettoyer_timbre(infos.get("timbre"))
            if timbre:
                resultat[code] = "|".join([resultat[code], *(f"{k}={v}" for k, v in sorted(timbre.items()))])
    return resultat


def _scores_echantillon(job_id: str, code: str, source, phrases: list[dict], pile, modele: dict) -> dict[int, float]:
    """Ressemblance de chaque phrase candidate avec la voix type du locuteur (pipeline/ressemblance.py)."""
    from pipeline.ressemblance import modele_empreinte, scores_candidats, seuil_echantillon

    if "empreinte" not in modele:
        modele["empreinte"] = pile.enter_context(modele_empreinte())
    if modele["empreinte"] is None:
        return {}
    scores = scores_candidats(source, phrases, modele["empreinte"])
    if scores:
        ecartees = sum(1 for v in scores.values() if v < seuil_echantillon(scores))
        logger.info("[%s] Échantillon de %s : %d phrase(s) évaluée(s), %d écartée(s) (voix différente)",
                    job_id, code, len(scores), ecartees)
    return scores


def _verifier_ressemblance(job_id: str, segments: list[dict], par_locuteur: dict[str, str], voix: dict[int, str],
                           langue: str, dossier, report: Callable[[float], None]) -> dict[int, float]:
    """Mesure la ressemblance de chaque phrase clonée avec l'échantillon du locuteur ; celles dont le timbre
    a dérivé sont refaites une fois en mode stable, et la meilleure des deux versions est gardée.

    Renvoie la ressemblance de chaque phrase mesurée. Les phrases refaites reçoivent « stabiliser »."""
    from pathlib import Path

    import numpy as np

    from pipeline.ressemblance import _lire, modele_empreinte, phrases_a_refaire, ressemblance_fichier
    from pipeline.voice import VoiceError, fichier_brut, generer_voix

    # Voix réellement clonées, sans transformation volontaire (âge, timbre réglé à la main)
    clones = {code: Path(spec[4:]) for code, spec in par_locuteur.items() if spec.startswith("ref:") and "|" not in spec}
    if not clones:
        return {}
    ressemblances: dict[int, float] = {}
    references: dict[str, object] = {}
    reprises: list[dict] = []
    with modele_empreinte() as empreinte:
        if empreinte is None:
            return {}
        for code, chemin in clones.items():
            reference = empreinte(_lire(chemin)) if chemin.is_file() else None
            if reference is None:
                continue
            references[code] = reference
            phrases = [s for s in segments if jobs.locuteur(s) == code]
            scores = {}
            for seg in phrases:
                r = ressemblance_fichier(fichier_brut(dossier, seg["id"]), reference, empreinte)
                if r is not None:
                    scores[seg["id"]] = r
            ressemblances.update(scores)
            refaire = phrases_a_refaire(scores)
            reprises += [s for s in phrases if s["id"] in refaire and not s.get("stabiliser")]  # une seule reprise
            if scores:
                logger.info("[%s] Ressemblance de %s : médiane %.2f, %d phrase(s) à refaire", job_id, code,
                            float(np.median(list(scores.values()))), len(refaire))
    report(0.2)
    if not reprises:
        return ressemblances

    avant = {s["id"]: dossier / f"avant_{s['id']:05d}.wav" for s in reprises}
    for s in reprises:
        fichier_brut(dossier, s["id"]).replace(avant[s["id"]])
        s["stabiliser"] = True
    try:
        try:
            generer_voix(reprises, langue, voix, dossier, on_progress=_sous_progression(report, 0.2, 0.9))
        except VoiceError as exc:
            logger.warning("[%s] Reprise des phrases au timbre dérivé impossible (%s) : versions d'origine gardées",
                           job_id, exc)
            return ressemblances
        with modele_empreinte() as empreinte:
            for s in reprises:
                nouvelle = None
                if empreinte is not None:
                    nouvelle = ressemblance_fichier(fichier_brut(dossier, s["id"]), references[jobs.locuteur(s)],
                                                    empreinte)
                if nouvelle is not None and nouvelle >= ressemblances[s["id"]]:
                    ressemblances[s["id"]] = nouvelle
                    avant[s["id"]].unlink()
        gardees = sum(1 for f in avant.values() if not f.is_file())
        logger.info("[%s] %d phrase(s) refaite(s), %d plus ressemblante(s) gardée(s)", job_id, len(reprises), gardees)
    finally:
        for ident, fichier in avant.items():  # version d'origine meilleure (ou reprise interrompue) : remise
            if fichier.is_file():
                fichier.replace(fichier_brut(dossier, ident))
    return ressemblances


def _verifier_ecoute(job_id: str, segments: list[dict], voix: dict[int, str], langue: str, dossier,
                     nouvelles: set[int], report: Callable[[float], None]) -> dict[int, float]:
    """Écoute de contrôle (pipeline/ecoute.py) des phrases générées à l'instant : chacune est retranscrite et
    comparée à son texte ; celles qui ne le disent pas (mots sautés, déformés, inventés) sont refaites une fois
    en mode stable, et la version la plus fidèle des deux est gardée. Renvoie la fidélité de chaque phrase."""
    from pipeline.ecoute import mesurer, modele_ecoute, phrases_a_refaire
    from pipeline.voice import VoiceError, fichier_brut, generer_voix

    a_ecouter = [s for s in segments if s["id"] in nouvelles]
    if not a_ecouter:
        return {}
    textes = {s["id"]: s["traduction"] for s in a_ecouter}
    fichiers = {s["id"]: fichier_brut(dossier, s["id"]) for s in a_ecouter}
    with modele_ecoute() as ecouter:
        if ecouter is None:
            return {}
        fidelites = mesurer(fichiers, textes, langue, ecouter, _sous_progression(report, 0, 0.4))
    refaire = phrases_a_refaire(fidelites)
    logger.info("[%s] Écoute de contrôle : %d phrase(s) écoutée(s), %d à refaire", job_id, len(fidelites), len(refaire))
    report(0.4)
    if not refaire:
        return fidelites

    reprises = [s for s in a_ecouter if s["id"] in refaire]
    avant = {s["id"]: dossier / f"avant_ecoute_{s['id']:05d}.wav" for s in reprises}
    for s in reprises:
        fichiers[s["id"]].replace(avant[s["id"]])
        s["stabiliser"] = True  # diction plus sage, sans intonation importée : moins de mots avalés
    try:
        try:
            generer_voix(reprises, langue, voix, dossier, on_progress=_sous_progression(report, 0.4, 0.85))
        except VoiceError as exc:
            logger.warning("[%s] Reprise des phrases infidèles impossible (%s) : versions d'origine gardées",
                           job_id, exc)
            return fidelites
        with modele_ecoute() as ecouter:
            nouvelles_fidelites = mesurer({i: fichiers[i] for i in refaire}, textes, langue, ecouter) if ecouter else {}
        for ident, valeur in nouvelles_fidelites.items():
            if valeur > fidelites[ident]:  # la reprise dit mieux son texte : on la garde
                fidelites[ident] = valeur
                avant[ident].unlink()
        gardees = sum(1 for f in avant.values() if not f.is_file())
        logger.info("[%s] %d phrase(s) refaite(s) après écoute, %d plus fidèle(s) gardée(s)", job_id, len(reprises),
                    gardees)
    finally:
        for ident, fichier in avant.items():  # version d'origine plus fidèle (ou reprise interrompue) : remise
            if fichier.is_file():
                fichier.replace(fichiers[ident])
    return fidelites


def _do_voices(job_id: str, report: Callable[[float], None]) -> None:
    """Génère chaque phrase traduite avec la voix de son locuteur (seules les phrases modifiées sont refaites)."""
    from pipeline.voice import Manifeste, fichier_brut, generer_voix

    status = jobs.read_status(job_id)
    data = jobs.read_segments(job_id)
    if any("ton" not in s for s in data["segments"]):
        _analyser_tons(job_id, data)
        jobs.write_segments(job_id, data)
    par_locuteur = voix_des_locuteurs(job_id, data)
    if (status.get("options") or {}).get("intonation_originale", True):
        _ajouter_intonation(job_id, status, data, par_locuteur)
    report(0.02)
    voix = {s["id"]: par_locuteur[jobs.locuteur(s)] for s in data["segments"]}
    dossier = jobs.project_dir(job_id) / "voix"
    deja = Manifeste(dossier).donnees
    for seg in data["segments"]:  # texte corrigé depuis : la phrase repart en mode normal (expressif)
        if seg.get("stabiliser") and deja.get(str(seg["id"]), {}).get("texte") != seg["traduction"]:
            seg.pop("stabiliser")
    def empreinte_fichier(ident: int):
        fichier = fichier_brut(dossier, ident)
        return fichier.stat().st_mtime_ns if fichier.is_file() else None

    avant = {s["id"]: empreinte_fichier(s["id"]) for s in data["segments"]}
    generees = generer_voix(data["segments"], status["langue_cible"], voix, dossier,
                            on_progress=_sous_progression(report, 0.02, 0.75))
    # Chaque voix clonée doit ressembler à son locuteur : les phrases au timbre dérivé sont refaites
    ressemblances = _verifier_ressemblance(job_id, data["segments"], par_locuteur, voix, status["langue_cible"],
                                           dossier, _sous_progression(report, 0.75, 0.85))
    # Chaque phrase générée à l'instant doit dire son texte : écoute de contrôle, reprise si besoin
    nouvelles = {s["id"] for s in data["segments"] if empreinte_fichier(s["id"]) != avant[s["id"]]}
    fidelites = _verifier_ecoute(job_id, data["segments"], voix, status["langue_cible"], dossier, nouvelles,
                                 _sous_progression(report, 0.85, 1.0))
    stabilisees = {s["id"] for s in data["segments"] if s.get("stabiliser")}
    # Voix réellement utilisée pour chaque locuteur : l'interface signale un changement non généré
    data = jobs.read_segments(job_id)
    for seg in data["segments"]:
        if seg["id"] in ressemblances:
            seg["ressemblance"] = ressemblances[seg["id"]]
        if seg["id"] in fidelites:
            seg["fidelite"] = fidelites[seg["id"]]
        elif seg["id"] in nouvelles:
            seg.pop("fidelite", None)  # nouvelle voix non écoutée (phrase trop courte) : ancienne mesure caduque
        if seg["id"] in stabilisees:
            seg["stabiliser"] = True  # gardé : la phrase reste « à jour » pour le cache des voix
        else:
            seg.pop("stabiliser", None)
    fiche = jobs.locuteurs(data)
    for infos in fiche.values():
        infos["voix_generee"] = infos["voix"]
        infos["timbre_genere"] = _timbre_du_locuteur(infos)
    data["locuteurs"] = fiche
    jobs.write_segments(job_id, data)
    jobs.update_status(job_id, voix_generees=generees)
    # Débit réel des voix : les prochaines estimations de durée (trop longue, trop courte) en tiennent compte
    from pipeline.calibrage import apprendre, mesurer_projet

    try:
        apprendre(status["langue_cible"], mesurer_projet(data["segments"], status["langue_cible"], dossier))
    except Exception:  # le calibrage ne doit jamais faire échouer le doublage
        logger.exception("[%s] Calibrage du débit des voix", job_id)


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


def _part_hauteur(phrases: list[dict], f0_locuteur: float | None, dossier, reference, egaliseur: str | None) -> float:
    """Part de la correction de hauteur gardée pour un locuteur cloné (pipeline/hauteur.py) : la plus forte qui
    ne rend pas la voix moins ressemblante au personnage (empreinte vocale, sur ses phrases les plus longues),
    ou 0. Sans le modèle d'empreinte, 0 : on ne transpose pas à l'aveugle."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    from pipeline.extract import _require
    from pipeline.hauteur import PARTS, TOLERANCE_RESSEMBLANCE, cible, decalage, filtre, hauteur_mediane
    from pipeline.ressemblance import _lire, modele_empreinte, ressemblance_fichier
    from pipeline.voice import fichier_brut

    longues = sorted((s for s in phrases if fichier_brut(dossier, s["id"]).is_file()),
                     key=lambda s: s["fin"] - s["debut"], reverse=True)[:4]
    ecarts = {s["id"]: decalage(hauteur_mediane(fichier_brut(dossier, s["id"])), cible(s, f0_locuteur)) for s in longues}
    if not any(ecarts.values()) or reference is None or not Path(reference).is_file():
        return 0.0
    with modele_empreinte() as empreinte, tempfile.TemporaryDirectory(prefix="voxlingo-hauteur-") as temp:
        if empreinte is None or (visee := empreinte(_lire(Path(reference)))) is None:
            return 0.0

        def ressemblance(part: float) -> float | None:
            scores = []
            for s in longues:
                brut = fichier_brut(dossier, s["id"])
                filtres = [f for f in (filtre(ecarts[s["id"]] * part if abs(ecarts[s["id"]] * part) >= 0.5 else 0.0),
                                       egaliseur) if f]
                sortie = Path(temp) / f"{part}_{brut.name}"
                if filtres:
                    subprocess.run([_require("ffmpeg"), "-y", "-v", "error", "-i", str(brut), "-af", ",".join(filtres),
                                    str(sortie)], capture_output=True)
                else:
                    sortie = brut
                score = ressemblance_fichier(sortie, visee, empreinte)
                if score is not None:
                    scores.append(score)
            return float(np.median(scores)) if scores else None

        sans = ressemblance(0.0)
        if sans is None:
            return 0.0
        for part in PARTS:
            avec = ressemblance(part)
            if avec is not None and avec >= sans - TOLERANCE_RESSEMBLANCE:
                return part
    return 0.0


def _egalisation_sure(phrases: list[dict], filtre: str, dossier, reference) -> bool:
    """La voix égalisée ressemble-t-elle au moins autant au locuteur que la voix brute ? (empreinte vocale,
    sur ses phrases les plus longues). Sans le modèle d'empreinte, l'égalisation est gardée."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    from pipeline.extract import _require
    from pipeline.ressemblance import _lire, modele_empreinte, ressemblance_fichier
    from pipeline.voice import fichier_brut

    longues = sorted((s for s in phrases if fichier_brut(dossier, s["id"]).is_file()),
                     key=lambda s: s["fin"] - s["debut"], reverse=True)[:4]
    if not longues or not Path(reference).is_file():
        return True
    with modele_empreinte() as empreinte, tempfile.TemporaryDirectory(prefix="voxlingo-timbre-") as temp:
        if empreinte is None:
            return True
        cible = empreinte(_lire(Path(reference)))
        if cible is None:
            return True
        brutes, egalisees = [], []
        for s in longues:
            brut = fichier_brut(dossier, s["id"])
            egalise = Path(temp) / brut.name
            subprocess.run([_require("ffmpeg"), "-y", "-v", "error", "-i", str(brut), "-af", filtre, str(egalise)],
                           capture_output=True)
            a, b = ressemblance_fichier(brut, cible, empreinte), ressemblance_fichier(egalise, cible, empreinte)
            if a is not None and b is not None:
                brutes.append(a)
                egalisees.append(b)
    return not brutes or float(np.median(egalisees)) >= float(np.median(brutes)) - 0.01


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

    # Fidélité à la voix d'origine : volume au décibel près, et timbre rapproché de la voix d'origine du
    # locuteur (pipeline/timbre_eq.py : mesuré de 100 Hz à 6,3 kHz, là où la voix isolée est fidèle) ; la
    # correction n'est gardée que si la voix ressemble au moins autant au locuteur (_egalisation_sure).
    from config import get_settings
    from pipeline.niveau import lire_extrait, niveau_actif_db
    from pipeline.timbre_eq import correction_locuteur, filtre_ffmpeg
    from pipeline.voice import fichier_reference

    voix_isolee = folder / "voix_isolee.wav"
    egaliseurs: dict[str, str | None] = {}

    def reference_du_locuteur(code: str):
        """Échantillon de clonage actuel, ou celui d'une version précédente (voix pas encore refaites)."""
        actuel = fichier_reference(dossier, code)
        # « reference_v3_X.wav » passe avant « reference_X.wav » (première version, sans numéro)
        return actuel if actuel.is_file() else max(dossier.glob(f"reference*_{code}.wav"), default=None)

    if voix_isolee.is_file() and get_settings().egalisation_timbre:
        for code, infos in jobs.locuteurs(data).items():
            phrases = [s for s in data["segments"] if jobs.locuteur(s) == code]
            clone = (infos.get("voix_generee", infos.get("voix")) == "clone"
                     and sum(s["fin"] - s["debut"] for s in phrases) >= REFERENCE_MIN_LOCUTEUR_S
                     and reference_du_locuteur(code) is not None)
            if not clone:
                continue  # voix de la bibliothèque : son timbre est choisi, pas recopié
            gains = correction_locuteur(voix_isolee, phrases, [fichier_brut(dossier, s["id"]) for s in phrases])
            filtre = filtre_ffmpeg(gains)
            if filtre and not _egalisation_sure(phrases, filtre, dossier, reference_du_locuteur(code)):
                logger.info("[%s] Timbre de %s : correction écartée (la voix ressemblerait moins au locuteur)",
                            job_id, code)
                filtre = None
            egaliseurs[code] = filtre
            if filtre:
                logger.info("[%s] Timbre de %s : correction %s dB", job_id, code,
                            " ".join(f"{g:+.0f}" for g in gains))
    report(0.05)

    # Hauteur des voix clonées calée sur le personnage (pipeline/hauteur.py) ; une voix de la bibliothèque
    # garde la hauteur choisie
    from pipeline.hauteur import cible as hauteur_cible

    hauteurs_locuteurs = {code: infos.get("f0_hz") for code, infos in jobs.locuteurs(data).items()
                          if infos.get("voix_generee", infos.get("voix")) == "clone"}
    # Part de la correction jugée à l'empreinte vocale : jamais une voix moins ressemblante au personnage
    parts_hauteur = {code: _part_hauteur([s for s in data["segments"] if jobs.locuteur(s) == code], f0,
                                         dossier, reference_du_locuteur(code), egaliseurs.get(code))
                     for code, f0 in hauteurs_locuteurs.items()}
    for code, part in parts_hauteur.items():
        logger.info("[%s] Hauteur de %s : %s", job_id, code,
                    f"{part:.0%} de la correction" if part else "gardée (la voix ressemblerait moins au personnage)")
    placements = []
    for n, seg in enumerate(data["segments"]):
        cale = dossier / f"cale_{seg['id']:05d}.wav"
        niveau = niveau_actif_db(*lire_extrait(voix_isolee, seg["debut"], seg["fin"])) if voix_isolee.is_file() else None
        resultat = caler(fichier_brut(dossier, seg["id"]), cale, seg, gain_volume(seg),
                         egaliseurs.get(jobs.locuteur(seg)), niveau,
                         voix_isolee if voix_isolee.is_file() else None,
                         hauteur_cible(seg, hauteurs_locuteurs[jobs.locuteur(seg)])
                         if jobs.locuteur(seg) in hauteurs_locuteurs else None,
                         parts_hauteur.get(jobs.locuteur(seg), 0.0))
        seg["hauteur_dt"] = resultat["hauteur_dt"]
        seg["niveau_origine_db"] = niveau
        seg["gain_db"] = resultat["gain_db"]
        seg["facteur_voix"] = resultat["facteur"]
        seg["duree_doublage"] = resultat["duree"]
        seg["trop_long"] = resultat["trop_long"]  # mesure réelle, remplace l'estimation
        placements.append((seg["debut"], cale))
        report(0.05 + 0.45 * (n + 1) / len(data["segments"]))
    jobs.write_segments(job_id, data)

    construire_piste(placements, duree, folder / "voix_doublee.wav")
    # Même acoustique que la scène : la voix doublée résonne comme la voix d'origine (pas de voix « de studio »)
    status["piece"] = None
    if voix_isolee.is_file() and get_settings().acoustique_piece:
        from pipeline.acoustique import appliquer_piece, choisir_piece

        piece = choisir_piece(voix_isolee, folder / "voix_doublee.wav")
        if piece is not None:
            avec_piece = folder / "voix_doublee_piece.wav"
            appliquer_piece(folder / "voix_doublee.wav", avec_piece, piece["reverberation_s"], piece["quantite_db"])
            avec_piece.replace(folder / "voix_doublee.wav")
            status["piece"] = {k: round(v, 2) for k, v in piece.items()}
    jobs.update_status(job_id, piece=status["piece"])
    report(0.6)
    options = status.get("options") or {}
    if options.get("conserver_fond", True):
        from pipeline.separate import separer

        from pipeline.assemble import residu_hors_parole

        voix_origine, fond = separer(folder / status["fichier"], folder)  # projets créés avant la phase 5
        # Sons d'origine pris pour de la voix par la séparation (rires, cris, bruitages), rendus au fond
        # là où personne ne parle
        residu = residu_hors_parole(voix_origine, [(s["debut"], s["fin"]) for s in data["segments"]],
                                    folder / "residu_fond.wav")
        mixer_fond(folder / "voix_doublee.wav", fond, int(options.get("volume_fond", 100)), folder / "doublage.wav",
                   residu)
        residu.unlink(missing_ok=True)
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
    invalider_habillage(job_id)  # la vidéo de base va changer
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
    _habiller_si_termine(job_id)


def _habiller_si_termine(job_id: str) -> None:
    """Après un export réussi : habillage avec les réglages du projet (le projet est déjà « terminé »,
    l'écran Résultat s'affiche et suit la progression de l'habillage)."""
    if (jobs.read_status(job_id) or {}).get("etat") == "termine":
        habiller_projet(job_id)


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


def run_remix(job_id: str) -> None:
    """Refait le mixage du son avec les voix déjà générées (calage, timbre, volume, fond sonore), sans les
    régénérer. Une vidéo synchronisée aux lèvres reçoit le nouveau son sans être recalculée."""
    from pipeline.format import apercu_navigateur, exporter_meme_format

    invalider_habillage(job_id)

    def body() -> None:
        _run_step(job_id, "export", lambda report: _do_export(job_id, report))
        status = jobs.read_status(job_id) or {}
        folder = jobs.project_dir(job_id)
        synchro = next(folder.glob("doublage_synchro.*"), None) if status.get("synchro_importee") else None
        if synchro is not None:
            remise = synchro.with_name(f"synchro_remixee{synchro.suffix}")
            exporter_meme_format(synchro, folder / "doublage.wav", remise)  # image copiée, nouveau son
            remise.replace(synchro)
            apercu_navigateur(synchro, folder / "apercu_synchro.mp4")

    _guarded(job_id, body, etat_final="termine")
    _habiller_si_termine(job_id)


def run_lipsync(job_id: str) -> None:
    """Synchronisation labiale demandée après coup depuis l'écran Résultat (moteur sur GPU)."""
    jobs.update_status(job_id, synchro_erreur=None)
    _guarded(job_id, lambda: _synchro_labiale(job_id), etat_final="termine")
    _habiller_si_termine(job_id)


# ---------------------------------------------------------------------------------------------------
# Habillage (sous-titres incrustés, flou des anciens sous-titres, logo, qualité, métadonnées)
# ---------------------------------------------------------------------------------------------------

# Chaque demande d'habillage reçoit un numéro : un rendu dont le numéro n'est plus le dernier s'arrête
# (réglages changés, voix régénérées…), et un seul rendu par projet tourne à la fois.
_habillage_numero: dict[str, int] = {}
_habillage_verrous: dict[str, threading.Lock] = {}
_habillage_garde = threading.Lock()


def _nouvelle_demande_habillage(job_id: str) -> tuple[int, threading.Lock]:
    with _habillage_garde:
        _habillage_numero[job_id] = _habillage_numero.get(job_id, 0) + 1
        return _habillage_numero[job_id], _habillage_verrous.setdefault(job_id, threading.Lock())


def video_de_base(job_id: str, status: dict):
    """Vidéo à habiller : la vidéo synchronisée si elle existe, sinon la vidéo doublée."""
    folder = jobs.project_dir(job_id)
    if status.get("synchro_importee"):
        synchro = next(folder.glob("doublage_synchro.*"), None)
        if synchro is not None:
            return synchro
    sortie = folder / (status.get("fichier_sortie") or "")
    return sortie if status.get("fichier_sortie") and sortie.is_file() else None


def _supprimer_habillage(job_id: str) -> None:
    folder = jobs.project_dir(job_id)
    for fichier in [*folder.glob("habille.*"), folder / "apercu_habille.mp4"]:
        fichier.unlink(missing_ok=True)


def invalider_habillage(job_id: str) -> None:
    """La vidéo de base va changer : le rendu en cours s'arrête et l'ancienne vidéo habillée est retirée."""
    numero, verrou = _nouvelle_demande_habillage(job_id)
    with verrou:
        _supprimer_habillage(job_id)
        jobs.update_status(job_id, habillage=None)


def habiller_projet(job_id: str) -> None:
    """Refait la vidéo habillée avec les réglages du projet (options « habillage »), sans toucher aux voix.

    Un échec n'affecte pas le doublage : il est signalé dans `habillage.erreur`, la vidéo doublée reste
    téléchargeable telle quelle."""
    from pipeline.format import apercu_navigateur
    from pipeline.habillage import actif, habiller, normaliser

    numero, verrou = _nouvelle_demande_habillage(job_id)
    with verrou:
        if _habillage_numero.get(job_id) != numero:
            return  # une demande plus récente passera après
        status = jobs.read_status(job_id) or {}
        reglages = (status.get("options") or {}).get("habillage")
        _supprimer_habillage(job_id)
        base = video_de_base(job_id, status)
        if not actif(reglages) or base is None:
            jobs.update_status(job_id, habillage=None)
            return
        folder = jobs.project_dir(job_id)
        sortie = folder / f"habille{base.suffix.lower()}"
        jobs.update_status(job_id, habillage={"etat": "en_cours", "progression": 0, "erreur": None})
        dernier = {"pourcent": 0}

        def progression(fraction: float) -> None:
            pourcent = int(fraction * 95)  # les 5 % restants : copie pour l'aperçu dans le navigateur
            if pourcent != dernier["pourcent"]:
                dernier["pourcent"] = pourcent
                jobs.update_status(job_id, habillage={"etat": "en_cours", "progression": pourcent, "erreur": None})

        debut = time.perf_counter()
        try:
            segments = jobs.read_segments(job_id)["segments"]
            habiller(base, segments, reglages, sortie, progression,
                     doit_arreter=lambda: _habillage_numero.get(job_id) != numero)
            apercu_navigateur(sortie, folder / "apercu_habille.mp4")
        except InterruptedError:
            logger.info("[%s] Habillage interrompu par une demande plus récente", job_id)
            return
        except Exception as exc:
            logger.exception("[%s] Habillage", job_id)
            _supprimer_habillage(job_id)
            jobs.update_status(job_id, habillage={"etat": "erreur", "progression": 0,
                                                  "erreur": str(exc) or exc.__class__.__name__})
            return
        logger.info("[%s] Habillage terminé en %.1f s", job_id, time.perf_counter() - debut)
        jobs.update_status(job_id, habillage={
            "etat": "termine", "progression": 100, "erreur": None, "fichier": sortie.name,
            "source": base.name, "reglages": normaliser(reglages),
        })
    sauvegarde.apres_export(job_id)
