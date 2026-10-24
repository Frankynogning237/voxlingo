"""Calage de la voix doublée sur les pauses de la voix d'origine, à l'intérieur d'une phrase.

Une phrase d'origine de plusieurs secondes contient des silences : le personnage s'arrête, respire,
écoute. La voix générée, elle, enchaîne d'un bloc : calée d'une seule pièce, elle parle pendant ces
silences (bouche fermée) puis reste en retard pour la suite. Ici, la voix générée est coupée à ses
propres respirations, et chaque morceau démarre quand le personnage reprend la parole :
  1. passages parlés d'origine : mesurés dans la voix d'origine isolée (énergie), regroupés tant que la
     pause est courte ; à défaut, les mots minutés par Whisper (moins fiables : il étire les mots sur les
     silences) ;
  2. respirations de la voix générée : silences d'au moins RESPIRATION_MIN_S (énergie) ;
  3. appariement (programmation dynamique) : chaque pause d'origine reçoit la respiration placée au même
     endroit du texte (même part de parole écoulée), ou est sautée si aucune ne convient : le coût est,
     dans les deux cas, le temps pendant lequel la voix parlerait bouche fermée (ou se tairait bouche ouverte) ;
  4. chaque morceau est posé au début de son passage, accéléré ou ralenti dans les mêmes limites que le
     calage d'un bloc (pipeline/align.py) pour le remplir ;
  5. chaque morceau suit l'intensité de son passage d'origine (le personnage hausse le ton, puis baisse la
     voix), à ±NUANCE_MAX_DB près autour du volume de la phrase, calé ensuite au décibel près ; de courts
     fondus évitent tout clic aux jonctions.
Une phrase d'un seul passage dont le personnage ne parle qu'après un silence (Whisper fait souvent
commencer la phrase trop tôt) est posée au moment où il commence vraiment à parler.
Sinon (pas de pause, pas de décalage), rien n'est fait : calage d'un bloc.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pipeline.extract import _require

PAUSE_INTERNE_S = 0.35  # silence d'origine à respecter (en dessous : simple liaison entre deux mots)
RESPIRATION_MIN_S = 0.12  # silence de la voix générée où l'on peut couper
TRAME_S = 0.02
MARGE_MORCEAU_S = 0.04  # gardé de part et d'autre d'un morceau (attaques et fins de mots)
ECART_MIN_S = 0.05  # entre deux morceaux qui se suivent
PASSAGE_MIN_S = 0.08  # bruit bref de la voix d'origine, ignoré
DECALAGE_MIN_S = 0.15  # silence d'origine en début de phrase à partir duquel la voix est décalée
ACCELERATION_MAX = 1.25
RALENTISSEMENT_MAX = 0.85
NUANCE_MAX_DB = 6.0  # écart d'intensité maximal d'un morceau par rapport au reste de la phrase
FONDU_S = 0.01
TOLERANCE_FIN = 1.12  # légère accélération d'ensemble si les morceaux dépassent un peu le temps disponible
ACCELERATION_LIMITE = 1.3  # aucun morceau au-delà, accélération d'ensemble comprise


def passages_origine(seg: dict[str, Any], pause_min: float = PAUSE_INTERNE_S) -> list[tuple[float, float]]:
    """Passages parlés de la phrase d'origine (secondes depuis son début), d'après ses mots minutés."""
    mots = [m for m in seg.get("mots") or [] if m.get("fin", 0) > m.get("debut", 0)]
    passages: list[list[float]] = []
    for m in mots:
        a, b = m["debut"] - seg["debut"], m["fin"] - seg["debut"]
        if passages and a - passages[-1][1] < pause_min:
            passages[-1][1] = max(passages[-1][1], b)
        else:
            passages.append([max(0.0, a), b])
    return [(a, b) for a, b in passages]


def passages_audio(origine: Path, seg: dict[str, Any],
                   pause_min: float = PAUSE_INTERNE_S) -> list[tuple[float, float]] | None:
    """Passages parlés de la phrase, mesurés dans la voix d'origine isolée (secondes depuis son début)."""
    try:
        with sf.SoundFile(origine) as f:
            taux = f.samplerate
            f.seek(min(int(seg["debut"] * taux), f.frames))
            extrait = f.read(max(0, int((seg["fin"] - seg["debut"]) * taux)), dtype="float32", always_2d=True)
    except (RuntimeError, OSError):
        return None
    actif = activite(extrait.mean(axis=1), taux)
    if not actif.any():
        return None
    passages: list[list[float]] = []
    for i in np.flatnonzero(actif):
        a, b = i * TRAME_S, (i + 1) * TRAME_S
        if passages and a - passages[-1][1] < pause_min:
            passages[-1][1] = b
        else:
            passages.append([a, b])
    return [(a, b) for a, b in passages if b - a >= PASSAGE_MIN_S] or None


def activite(voix: np.ndarray, taux: int) -> np.ndarray:
    """Trames de TRAME_S où la voix parle (énergie à moins de 30 dB de ses passages forts)."""
    pas = max(1, int(TRAME_S * taux))
    n = len(voix) // pas
    if n == 0:
        return np.zeros(0, dtype=bool)
    energie = 10 * np.log10(np.mean(voix[: n * pas].reshape(n, pas) ** 2, axis=1) + 1e-10)
    return energie > max(float(np.percentile(energie, 95)) - 30.0, -55.0)


def respirations(actif: np.ndarray, minimum: float = RESPIRATION_MIN_S) -> list[tuple[float, float]]:
    """Silences internes de la voix générée (début, fin en secondes), assez longs pour y couper."""
    parle = np.flatnonzero(actif)
    if len(parle) == 0:
        return []
    silences, debut = [], None
    for i in range(parle[0], parle[-1] + 1):
        if not actif[i] and debut is None:
            debut = i
        elif actif[i] and debut is not None:
            if (i - debut) * TRAME_S >= minimum:
                silences.append((debut * TRAME_S, i * TRAME_S))
            debut = None
    return silences


def apparier(passages: list[tuple[float, float]], parole_generee: list[float]) -> list[tuple[int, int]]:
    """Choisit quelles pauses d'origine respecter, et à quelle respiration de la voix générée.

    `passages` : passages parlés d'origine ; `parole_generee` : part de parole déjà dite (0 à 1) à chaque
    respiration de la voix générée. Renvoie [(indice de pause d'origine, indice de respiration)], croissant
    des deux côtés. Coût en secondes : écart de position (pause respectée trop tôt ou trop tard) ou durée de
    la pause sautée (la voix y parlerait bouche fermée)."""
    pauses = len(passages) - 1
    if pauses < 1 or not parole_generee:
        return []
    durees = [b - a for a, b in passages]
    total = sum(durees) or 1.0
    deja = np.cumsum(durees)[:-1]  # parole d'origine écoulée à chaque pause
    sautee = [passages[i + 1][0] - passages[i][1] for i in range(pauses)]
    m = len(parole_generee)
    inf = float("inf")
    # meilleur[i][j] : coût des pauses 0..i-1 traitées, respirations 0..j-1 disponibles
    meilleur = np.full((pauses + 1, m + 1), inf)
    meilleur[0, :] = 0.0
    choix: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for i in range(1, pauses + 1):
        for j in range(0, m + 1):
            # pause i-1 sautée
            cout = meilleur[i - 1, j] + sautee[i - 1]
            meilleur[i, j], choix[i, j] = cout, (i - 1, j, False)
            # respiration j-1 attribuée à la pause i-1
            if j >= 1:
                cout = meilleur[i - 1, j - 1] + abs(parole_generee[j - 1] * total - deja[i - 1])
                if cout < meilleur[i, j]:
                    meilleur[i, j], choix[i, j] = cout, (i - 1, j - 1, True)
            # respiration j-1 ignorée
            if j >= 1 and meilleur[i, j - 1] < meilleur[i, j]:
                meilleur[i, j], choix[i, j] = meilleur[i, j - 1], choix[i, j - 1]
    paires, i, j = [], pauses, m
    while i > 0:
        pi, pj, pris = choix[i, j]
        if pris:
            paires.append((pi, pj))
        i, j = pi, pj
    return sorted(set(paires))


def plan(seg: dict[str, Any], voix: np.ndarray, taux: int,
         passages: list[tuple[float, float]] | None = None) -> list[dict[str, float]] | None:
    """Morceaux à poser : [{de, a (secondes dans la voix générée), pose, duree (dans la phrase)}], ou None
    si la phrase n'a pas de pause à respecter. `passages` : passages parlés d'origine (sinon, les mots)."""
    passages = passages or passages_origine(seg)
    actif = activite(voix, taux)
    if not passages or not actif.any():
        return None
    parle = np.flatnonzero(actif)
    debut_voix, fin_voix = parle[0] * TRAME_S, (parle[-1] + 1) * TRAME_S
    silences = respirations(actif)
    cumul = np.cumsum(actif) * TRAME_S
    parole = cumul[-1] or 1.0
    parts = [float(cumul[min(int(a / TRAME_S), len(cumul) - 1)]) / parole for a, _ in silences]
    paires = apparier(passages, parts) if silences else []
    if not paires and passages[0][0] < DECALAGE_MIN_S:
        return None  # ni pause à respecter, ni silence de départ : calage d'un bloc
    # Bornes des morceaux de voix générée et des groupes de passages d'origine qui leur correspondent
    coupes = [(silences[j][0] + silences[j][1]) / 2 for _, j in paires]
    bornes_voix = [debut_voix, *coupes, fin_voix]
    groupes, premier = [], 0
    for i, _ in paires:
        groupes.append((passages[premier][0], passages[i][1]))
        premier = i + 1
    groupes.append((passages[premier][0], passages[-1][1]))
    morceaux = []
    for k, (a, b) in enumerate(zip(bornes_voix, bornes_voix[1:])):
        de, a_ = max(0.0, a - MARGE_MORCEAU_S), min(len(voix) / taux, b + MARGE_MORCEAU_S)
        morceaux.append({"de": de, "a": a_, "pose": groupes[k][0], "duree": groupes[k][1] - groupes[k][0]})
    return morceaux


def _atempo(extrait: np.ndarray, taux: int, facteur: float) -> np.ndarray:
    if abs(facteur - 1.0) < 0.01:
        return extrait
    with tempfile.TemporaryDirectory(prefix="voxlingo-calage-") as temp:
        entree, sortie = Path(temp) / "e.wav", Path(temp) / "s.wav"
        sf.write(entree, extrait, taux)
        subprocess.run([_require("ffmpeg"), "-y", "-v", "error", "-i", str(entree), "-filter:a",
                        f"atempo={facteur:.4f}", str(sortie)], check=True, capture_output=True)
        return sf.read(sortie, dtype="float32")[0]


def niveau_actif(extrait: np.ndarray, taux: int) -> float | None:
    """Niveau (dB) des trames où la voix parle."""
    actif = activite(extrait, taux)
    if not actif.any():
        return None
    pas = max(1, int(TRAME_S * taux))
    trames = extrait[: len(actif) * pas].reshape(len(actif), pas)[actif]
    return float(10 * np.log10(np.mean(trames ** 2) + 1e-12))


def nuances(niveaux_origine: list[float | None], niveaux_voix: list[float | None],
            durees: list[float]) -> list[float]:
    """Gain (dB) de chaque morceau pour suivre l'intensité de son passage d'origine : relatif au reste de la
    phrase (moyenne pondérée nulle), plafonné à ±NUANCE_MAX_DB. Sans mesure, aucun gain."""
    ecarts = [o - v if o is not None and v is not None else None for o, v in zip(niveaux_origine, niveaux_voix)]
    mesures = [(e, d) for e, d in zip(ecarts, durees) if e is not None]
    if len(mesures) < 2:
        return [0.0] * len(durees)
    moyenne = sum(e * d for e, d in mesures) / (sum(d for _, d in mesures) or 1.0)
    return [float(np.clip(e - moyenne, -NUANCE_MAX_DB, NUANCE_MAX_DB)) if e is not None else 0.0 for e in ecarts]


def _fondus(x: np.ndarray, taux: int) -> np.ndarray:
    n = min(int(FONDU_S * taux), len(x) // 2)
    if n > 0:
        rampe = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= rampe
        x[-n:] *= rampe[::-1]
    return x


def _niveaux_origine(origine: Path | None, seg: dict[str, Any], morceaux: list[dict[str, float]]) -> list[float | None]:
    """Niveau de la voix d'origine sur chaque passage où un morceau est posé."""
    if origine is None:
        return [None] * len(morceaux)
    niveaux = []
    try:
        with sf.SoundFile(origine) as f:
            taux = f.samplerate
            for m in morceaux:
                f.seek(min(int((seg["debut"] + m["pose"]) * taux), f.frames))
                extrait = f.read(int(m["duree"] * taux), dtype="float32", always_2d=True).mean(axis=1)
                niveaux.append(niveau_actif(extrait, taux) if len(extrait) else None)
    except (RuntimeError, OSError):
        return [None] * len(morceaux)
    return niveaux


def caler_sur_pauses(brut: Path, seg: dict[str, Any], sortie: Path,
                     origine: Path | None = None) -> dict[str, float] | None:
    """Écrit la voix calée morceau par morceau sur les passages d'origine (`origine` : voix d'origine isolée).
    Renvoie {facteur, duree} (facteur moyen pondéré), ou None si la phrase n'a pas de pause à respecter ou si
    le résultat dépasserait le temps disponible : le calage d'un bloc s'applique alors."""
    voix, taux = sf.read(brut, dtype="float32")
    if voix.ndim > 1:
        voix = voix.mean(axis=1)
    morceaux = plan(seg, voix, taux, passages_audio(origine, seg) if origine is not None else None)
    if not morceaux:
        return None
    disponible = max(seg.get("fin_max", seg["fin"]) - seg["debut"], 0.1)
    extraits = [voix[int(m["de"] * taux): int(m["a"] * taux)] for m in morceaux]
    gains = nuances(_niveaux_origine(origine, seg, morceaux), [niveau_actif(x, taux) for x in extraits],
                    [m["duree"] for m in morceaux])
    pieces, position, poids, total, plus_rapide = [], 0.0, 0.0, 0.0, 1.0
    for m, extrait, gain in zip(morceaux, extraits, gains):
        extrait = _fondus(extrait * np.float32(10 ** (gain / 20)), taux)
        longueur = len(extrait) / taux
        facteur = min(max(longueur / max(m["duree"] + 2 * MARGE_MORCEAU_S, 0.05), RALENTISSEMENT_MAX),
                      ACCELERATION_MAX)
        pose = max(m["pose"] - MARGE_MORCEAU_S, position + ECART_MIN_S if pieces else 0.0)
        cale = _atempo(extrait, taux, facteur)
        pieces.append((pose, cale))
        position = pose + len(cale) / taux
        poids += longueur
        total += longueur / facteur
        plus_rapide = max(plus_rapide, facteur)
    ensemble = position / disponible
    if ensemble > TOLERANCE_FIN or plus_rapide * max(ensemble, 1.0) > ACCELERATION_LIMITE:
        return None
    sortie_audio = np.zeros(int(np.ceil(position * taux)) + 1, dtype=np.float32)
    for pose, cale in pieces:
        i = int(pose * taux)
        sortie_audio[i: i + len(cale)] += cale[: len(sortie_audio) - i]
    if ensemble > 1.0:  # dépasse un peu : tout est légèrement accéléré pour tenir
        sortie_audio = _atempo(sortie_audio, taux, ensemble)
        total /= ensemble
    sf.write(sortie, sortie_audio, taux)
    return {"facteur": round(poids / total, 3) if total else 1.0, "duree": round(len(sortie_audio) / taux, 2)}
