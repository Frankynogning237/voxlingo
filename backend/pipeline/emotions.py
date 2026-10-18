"""Émotion de chaque phrase : ce que la voix exprime, comment la phrase est dite, ce que montre le visage.

Trois sources, croisées :
  1. **la voix** : emotion2vec+ (modèle ouvert, entraîné sur de nombreuses langues) reconnaît
     colère, joie, tristesse, peur, surprise, dégoût ou neutre ;
  2. **la manière de dire**, comparée aux habitudes du même locuteur (pipeline/prosody.py) :
     crie, parle fort, chuchote, rapide, lent, voix animée, monocorde ;
  3. **le visage** de la personne qui parle (lecture des scènes, pipeline/vision.py) : sourit,
     fronce les sourcils, air surpris, air triste.

Le résultat (`seg["emotion"]`) guide la traduction (adapter le ton, pas seulement les mots) et
l'expressivité de la voix doublée. Une émotion corrigée à la main dans l'éditeur est conservée.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("voxlingo.emotions")

MODELE = "iic/emotion2vec_plus_base"
EMOTIONS = ("neutre", "joie", "colère", "tristesse", "peur", "surprise", "dégoût")
_ETIQUETTES = {
    "angry": "colère", "disgusted": "dégoût", "fearful": "peur", "happy": "joie", "neutral": "neutre",
    "other": "neutre", "sad": "tristesse", "surprised": "surprise", "<unk>": "neutre",
}
CONFIANCE_MIN = 0.5  # en dessous, la voix ne tranche pas : neutre (le visage peut préciser)
DUREE_MIN_S = 0.4


def _modele():
    from funasr import AutoModel

    from config import get_runtime

    return AutoModel(model=MODELE, hub="hf", disable_update=True, device=get_runtime().device)


def emotions_de_la_voix(segments: list[dict[str, Any]], source: Path,
                        on_progress: Callable[[float], None] | None = None) -> list[tuple[str, float] | None]:
    """(émotion, confiance) de chaque phrase d'après la voix, ou None si le modèle est indisponible."""
    import librosa

    try:
        modele = _modele()
    except Exception as exc:  # noqa: BLE001 — l'analyse de la manière de dire et du visage suffit
        logger.warning("Reconnaissance des émotions de la voix indisponible (%s)", exc)
        return [None] * len(segments)
    resultats: list[tuple[str, float] | None] = []
    try:
        for n, seg in enumerate(segments):
            duree = seg["fin"] - seg["debut"]
            if duree < DUREE_MIN_S:
                resultats.append(None)
                continue
            signal, _ = librosa.load(str(source), sr=16000, mono=True, offset=seg["debut"], duration=duree)
            sortie = modele.generate(signal, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
            etiquette, score = max(zip(sortie["labels"], sortie["scores"]), key=lambda kv: kv[1])
            resultats.append((_ETIQUETTES.get(etiquette.split("/")[-1], "neutre"), float(score)))
            if on_progress:
                on_progress((n + 1) / len(segments))
    finally:
        del modele
        gc.collect()
    return resultats


def _references(segments: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Façon habituelle de parler de chaque locuteur (médianes de sa prosodie)."""
    par_locuteur: dict[str, list[dict[str, Any]]] = {}
    for seg in segments:
        if seg.get("prosodie"):
            par_locuteur.setdefault(seg.get("locuteur") or "SPEAKER_00", []).append(seg["prosodie"])
    references = {}
    for code, mesures in par_locuteur.items():
        variations = [m["variation_dt"] for m in mesures if m.get("variation_dt") is not None]
        references[code] = {
            "energie_db": float(np.median([m["energie_db"] for m in mesures])),
            "variation_dt": float(np.median(variations)) if variations else 1.0,
            "debit": float(np.median([m["debit"] for m in mesures])) or 1.0,
        }
    return references


def maniere_de_dire(prosodie: dict[str, Any] | None, reference: dict[str, float] | None) -> list[str]:
    """Comment la phrase est dite, par rapport aux habitudes du même locuteur."""
    if not prosodie or not reference:
        return []
    maniere = []
    ecart = prosodie["energie_db"] - reference["energie_db"]
    if ecart > 8:
        maniere.append("crie")
    elif ecart > 4:
        maniere.append("parle fort")
    elif ecart < -10:
        maniere.append("chuchote")
    elif ecart < -5:
        maniere.append("parle doucement")
    if prosodie["debit"] > reference["debit"] * 1.25:
        maniere.append("rapide")
    elif prosodie["debit"] < reference["debit"] * 0.75:
        maniere.append("lent")
    variation = prosodie.get("variation_dt")
    if variation is not None:
        if variation > reference["variation_dt"] * 1.4:
            maniere.append("voix animée")
        elif variation < reference["variation_dt"] * 0.6:
            maniere.append("monocorde")
    return maniere


def expression_du_visage(vision: dict[str, Any] | None, perso: int | None, debut: float, fin: float) -> str | None:
    """Expression du personnage `perso` pendant [debut, fin] (formes MediaPipe médianes)."""
    if not vision or perso is None:
        return None
    ips = vision["ips"]
    a, b = int(debut * ips), int(np.ceil(fin * ips))
    valeurs = [e for p in vision["pistes"] if p["perso"] == perso and p.get("expressions")
               for n, e in zip(p["images"], p["expressions"], strict=True) if a <= n <= b]
    if len(valeurs) < 2:
        return None
    sourire, moue, froncement, sourcils_hauts, yeux = np.median(np.array(valeurs), axis=0)
    if yeux > 0.3 and sourcils_hauts > 0.3:
        return "air surpris"
    if sourire > 0.4:
        return "sourit"
    if froncement > 0.35:
        return "fronce les sourcils"
    if moue > 0.3 and sourcils_hauts > 0.2:
        return "air triste"
    return None


def _fusionner(voix: tuple[str, float] | None, maniere: list[str], visage: str | None) -> dict[str, Any]:
    emotion, confiance = voix if voix and voix[1] >= CONFIANCE_MIN else ("neutre", voix[1] if voix else 0.0)
    if emotion == "neutre":
        # La voix ne tranche pas : le visage peut préciser
        emotion = {"sourit": "joie", "air surpris": "surprise", "air triste": "tristesse"}.get(visage or "", "neutre")
    marque = sum(m in ("crie", "parle fort", "voix animée") for m in maniere)
    if emotion == "neutre":
        intensite = "légère"
    elif confiance >= 0.85 and marque:
        intensite = "forte"
    elif confiance >= 0.7 or marque:
        intensite = "moyenne"
    else:
        intensite = "légère"
    return {"emotion": emotion, "intensite": intensite, "confiance": round(confiance, 2),
            "maniere": maniere, "visage": visage}


def analyser(segments: list[dict[str, Any]], source: Path, vision: dict[str, Any] | None = None,
             locuteurs: dict[str, dict[str, Any]] | None = None,
             on_progress: Callable[[float], None] | None = None) -> None:
    """Ajoute `emotion` à chaque phrase (une émotion corrigée à la main est conservée)."""
    from pipeline.vision import qui_parle

    voix = emotions_de_la_voix(segments, source, on_progress)
    references = _references(segments)
    for seg, v in zip(segments, voix, strict=True):
        if (seg.get("emotion") or {}).get("corrigee"):
            continue
        code = seg.get("locuteur") or "SPEAKER_00"
        perso = None
        if vision:
            perso = qui_parle(vision, seg["debut"], seg["fin"])[0]
            if perso is None:
                perso = (locuteurs or {}).get(code, {}).get("personnage")
        seg["emotion"] = _fusionner(v, maniere_de_dire(seg.get("prosodie"), references.get(code)),
                                    expression_du_visage(vision, perso, seg["debut"], seg["fin"]))
    compte = {e: sum(1 for s in segments if (s.get("emotion") or {}).get("emotion") == e) for e in EMOTIONS}
    logger.info("Émotions : %s", {e: n for e, n in compte.items() if n})


def decrire(emotion: dict[str, Any] | None) -> str:
    """« colère (forte), crie, rapide ; visage : fronce les sourcils » pour la traduction."""
    if not emotion:
        return ""
    texte = emotion["emotion"] + (f" ({emotion['intensite']})" if emotion["emotion"] != "neutre" else "")
    if emotion.get("maniere"):
        texte += ", " + ", ".join(emotion["maniere"])
    if emotion.get("visage"):
        texte += f" ; visage : {emotion['visage']}"
    return texte
