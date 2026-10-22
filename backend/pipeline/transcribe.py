"""Transcription et détection de langue avec faster-whisper.

Le modèle est chargé pour la tâche puis déchargé, pour libérer la mémoire
avant les étapes suivantes. Le minutage de chaque mot est conservé : il sert
à redécouper précisément les phrases trop longues (pipeline/segments.py).

Pour une transcription fidèle :
  - la langue est détectée sur plusieurs passages parlés de la vidéo, pas seulement sur ses
    30 premières secondes (générique, musique, réplique dans une autre langue) ;
  - une courte phrase d'exemple ponctuée, dans la langue de la vidéo, amène Whisper à ponctuer :
    les phrases sont mieux délimitées, donc mieux traduites et mieux calées ;
  - les termes du glossaire du projet (noms propres, marques…) sont signalés à Whisper ;
  - les « hallucinations » de Whisper sont écartées : phrases inventées sur du silence ou de la
    musique (« Sous-titres réalisés par… », « Merci d'avoir regardé »), boucles de répétition.
"""

from __future__ import annotations

import gc
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.transcribe")


@dataclass
class Segment:
    id: int
    debut: float
    fin: float
    texte: str
    mots: list[dict] = field(default_factory=list)  # [{"mot", "debut", "fin"}]


@dataclass
class Transcription:
    langue: str
    probabilite: float
    duree_s: float
    segments: list[Segment]

    def to_dict(self) -> dict:
        return asdict(self)


FIN_DE_PHRASE = (".", "!", "?", "…", ":", ";", "。", "！", "？")
ECART_MAX_S = 0.6  # silence maximal entre deux morceaux d'une même phrase
DUREE_MAX_S = 15.0  # au-delà, on garde les segments séparés

# Dépôts des modèles : « large-v3-turbo » a changé d'adresse sur Hugging Face (mobiuslabsgmbh →
# dropbox-dash) et faster-whisper ne suit pas la redirection ; on essaie les dépôts dans l'ordre.
DEPOTS = {
    "large-v3-turbo": ["dropbox-dash/faster-whisper-large-v3-turbo", "deepdml/faster-whisper-large-v3-turbo-ct2",
                       "large-v3-turbo"],
    "turbo": ["dropbox-dash/faster-whisper-large-v3-turbo", "deepdml/faster-whisper-large-v3-turbo-ct2", "turbo"],
}

# Exemple ponctué dans la langue de la vidéo (court : Whisper le prend comme texte précédent, pas comme
# consigne), pour inciter à ponctuer. Mesuré sur une vidéo de 89 s parlée vite et sans pauses : sans lui,
# de longues suites sans ponctuation (« y a quoi leo me trompe avec nisqa qu'est ce que tu veux… ») ; avec
# lui, des phrases ponctuées et bien délimitées (« Léo me trompe avec Niska. »), pour ~30 % de temps en plus.
INVITE_PONCTUATION = True
INVITES = {
    "fr": "Bonjour. Alors, qu'est-ce que tu veux ? D'accord, c'est terminé !",
    "en": "Hello. So, what do you want? Okay, it's over!",
    "es": "Hola. Entonces, ¿qué quieres? ¡Vale, se acabó!",
    "de": "Hallo. Also, was willst du? Okay, es ist vorbei!",
    "it": "Ciao. Allora, cosa vuoi? Va bene, è finita!",
    "pt": "Olá. Então, o que você quer? Tudo bem, acabou!",
    "nl": "Hallo. Dus, wat wil je? Oké, het is voorbij!",
}

# Phrases que Whisper invente sur du silence ou de la musique (génériques de sous-titrage de ses données)
HALLUCINATIONS = re.compile(
    r"sous-titr|amara\.org|merci d'avoir regardé|abonnez-vous|thanks? (you )?for watching|subtitles by|"
    r"subscribe to|untertitel (im auftrag|der amara|von)|subtítulos (realizados|por)|legendas pela|"
    r"sottotitoli (creati|a cura)|ondertiteling",
    re.IGNORECASE,
)
NO_SPEECH_MAX = 0.6  # phrase écartée si Whisper la juge probablement non parlée…
LOGPROB_MIN = -1.0  # … et peu sûre
COMPRESSION_MAX = 2.6  # texte qui se répète en boucle (« oui oui oui oui… » sur 30 s)


def charger_modele(nom: str, device: str, compute_type: str):
    """Modèle Whisper, depuis le premier dépôt disponible (voir DEPOTS)."""
    from faster_whisper import WhisperModel

    racine = str(get_settings().models_dir / "whisper")
    erreur: Exception | None = None
    for depot in DEPOTS.get(nom, [nom]):
        try:
            return _reglages_audio_coherents(
                WhisperModel(depot, device=device, compute_type=compute_type, download_root=racine))
        except Exception as exc:  # noqa: BLE001 — dépôt déplacé ou injoignable : on essaie le suivant
            logger.warning("Modèle Whisper %s indisponible (%s)", depot, exc)
            erreur = exc
    raise RuntimeError(f"Impossible de charger le modèle de transcription {nom} : {erreur}")


def _reglages_audio_coherents(modele):
    """Si `preprocessor_config.json` manque (téléchargement interrompu), faster-whisper calcule 80 bandes de
    fréquences au lieu des 128 qu'attendent large-v3 et turbo, et la transcription échoue : on recale
    l'analyse du son sur ce que le modèle attend."""
    from faster_whisper.feature_extractor import FeatureExtractor

    attendues = getattr(modele.model, "n_mels", None)
    if attendues and modele.feature_extractor.mel_filters.shape[0] != attendues:
        logger.warning("Réglages audio du modèle Whisper incomplets : %d bandes de fréquences rétablies", attendues)
        modele.feat_kwargs = {**getattr(modele, "feat_kwargs", {}), "feature_size": attendues}
        modele.feature_extractor = FeatureExtractor(**modele.feat_kwargs)
    return modele


def fusionner_fragments(segments: list[Segment]) -> list[Segment]:
    """Recolle les phrases coupées par Whisper (« …into another » + « language. »).

    Traduire une phrase entière donne une traduction bien meilleure qu'un morceau isolé.
    """
    fusion: list[Segment] = []
    for seg in segments:
        prec = fusion[-1] if fusion else None
        if (
            prec is not None
            and not prec.texte.rstrip().endswith(FIN_DE_PHRASE)
            and seg.debut - prec.fin <= ECART_MAX_S
            and seg.fin - prec.debut <= DUREE_MAX_S
        ):
            prec.fin = seg.fin
            prec.texte = f"{prec.texte} {seg.texte}".strip()
            prec.mots.extend(seg.mots)
        else:
            fusion.append(Segment(id=len(fusion), debut=seg.debut, fin=seg.fin, texte=seg.texte,
                                  mots=list(seg.mots)))
    return fusion


def est_hallucination(texte: str, no_speech: float, logprob: float, compression: float,
                      precedent: str | None = None) -> bool:
    """Phrase que Whisper a probablement inventée : générique connu, silence jugé non parlé et transcrit
    sans assurance, boucle de répétition, ou même texte que la phrase précédente (Whisper qui bégaie)."""
    propre = texte.strip()
    if not propre:
        return True
    if len(propre) <= 90 and HALLUCINATIONS.search(propre):
        return True
    if no_speech > NO_SPEECH_MAX and logprob < LOGPROB_MIN:
        return True
    if compression > COMPRESSION_MAX:
        return True
    return precedent is not None and len(propre) > 12 and propre.casefold() == precedent.strip().casefold()


def termes_du_glossaire(glossaire: str) -> list[str]:
    """Termes de la langue d'origine du glossaire (« terme = traduction ») : noms propres, marques…"""
    from pipeline.translate import lire_glossaire

    return [terme for terme, _ in lire_glossaire(glossaire)][:30]


def _charger_16k(audio: Path):
    """Signal mono 16 kHz (float32) attendu par Whisper."""
    import librosa
    import numpy as np

    signal, _ = librosa.load(str(audio), sr=16000, mono=True)
    return signal.astype(np.float32)


def _detecter_langue(model, signal) -> tuple[str, float]:
    """Langue majoritaire sur plusieurs passages parlés (pas seulement les 30 premières secondes)."""
    try:
        langue, probabilite, _ = model.detect_language(signal, vad_filter=True, language_detection_segments=4)
        return langue, float(probabilite)
    except Exception as exc:  # noqa: BLE001 — ancienne version : détection faite par transcribe()
        logger.warning("Détection de langue sur plusieurs passages indisponible (%s)", exc)
        return "", 0.0


def transcribe(audio: Path, on_progress: Callable[[float], None] | None = None,
               termes: list[str] | None = None) -> Transcription:
    """Transcrit l'audio et détecte sa langue. `on_progress` reçoit une valeur entre 0 et 1 ; `termes`
    (glossaire du projet) aide Whisper à reconnaître les noms propres."""
    runtime = get_runtime()
    logger.info("Chargement de Whisper %s (%s, %s)", runtime.whisper_model, runtime.device,
                runtime.whisper_compute_type)
    model = charger_modele(runtime.whisper_model, runtime.device, runtime.whisper_compute_type)
    try:
        # Le son est lu ici (WAV via soundfile) plutôt que par faster-whisper, qui passe par PyAV :
        # certaines versions de PyAV (Kaggle) refusent son option « metadata_errors ».
        signal = _charger_16k(audio)
        langue, probabilite = _detecter_langue(model, signal)
        options = {}
        if termes:
            options["hotwords"] = ", ".join(termes)
        segments_iter, info = model.transcribe(
            signal, language=langue or None, vad_filter=True, beam_size=5, word_timestamps=True,
            initial_prompt=INVITES.get(langue) if INVITE_PONCTUATION else None, **options,
        )
        langue, probabilite = langue or info.language, probabilite or info.language_probability
        logger.info("Langue détectée : %s (%.0f %%)", langue, probabilite * 100)

        segments: list[Segment] = []
        ecartees = 0
        for seg in segments_iter:
            texte = seg.text.strip()
            if est_hallucination(texte, seg.no_speech_prob, seg.avg_logprob, seg.compression_ratio,
                                 segments[-1].texte if segments else None):
                ecartees += 1
                logger.info("Phrase écartée (inventée par Whisper ?) : %r", texte[:80])
                continue
            mots = [
                {"mot": w.word.strip(), "debut": round(w.start, 2), "fin": round(w.end, 2)}
                for w in (seg.words or [])
                if w.word.strip()
            ]
            segments.append(Segment(id=len(segments), debut=round(seg.start, 2), fin=round(seg.end, 2),
                                    texte=texte, mots=mots))
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))
        if ecartees:
            logger.info("%d phrase(s) écartée(s) comme inventées par Whisper", ecartees)

        return Transcription(
            langue=langue,
            probabilite=round(probabilite, 3),
            duree_s=round(info.duration, 2),
            segments=fusionner_fragments(segments),
        )
    finally:
        del model
        gc.collect()
        if runtime.device == "cuda":
            import torch

            torch.cuda.empty_cache()
