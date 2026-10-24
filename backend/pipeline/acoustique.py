"""Acoustique de la pièce : la voix doublée résonne comme la voix d'origine.

Une voix de synthèse est « sèche » : enregistrée comme en studio, elle se coupe net à la fin de chaque
phrase, alors que la voix d'origine, prise dans une vraie pièce, s'éteint progressivement (réverbération).
Posée telle quelle sur la scène, elle sonne « collée par-dessus » la vidéo. Ici :
  1. on mesure, dans la voix d'origine isolée, comment la voix s'éteint après chaque fin de phrase
     (courbe de chute, en dB, sur 200 ms) ;
  2. on essaie des pièces simulées (durée de réverbération, quantité de réverbération) sur des extraits de
     la voix doublée, et l'on garde celle dont la courbe de chute ressemble le plus à celle d'origine ;
  3. la pièce choisie est appliquée à toute la piste des voix doublées (convolution par blocs : mémoire
     bornée), à volume constant.
La voix d'origine isolée contient souvent un léger bruit de fond (restes de la séparation) qui masque la
fin de la réverbération : il est mesuré juste après les fins de phrase (de 300 à 400 ms), et les courbes ne
sont comparées qu'au-dessus de lui. Si la réverbération
d'origine n'est pas mesurable, ou si la voix doublée lui ressemble déjà, rien n'est appliqué.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, oaconvolve, sosfilt

logger = logging.getLogger("voxlingo.acoustique")

TRAME_S = 0.005
DUREE_COURBE_S = 0.2  # courbe de chute mesurée sur 200 ms après la fin de la parole
SILENCE_APRES_S = 0.4  # fin de phrase retenue : suivie d'au moins 0,4 s sans parole
BRUIT_S = (0.3, 0.4)  # où l'on mesure le bruit de fond d'origine, après une fin de phrase
AVANT_S = 0.1  # crête cherchée dans les 100 ms qui précèdent la fin
BANDE_HZ = (300, 3400)
FINS_MAX = 60  # extraits de la voix doublée utilisés pour essayer les pièces
REVERBERATIONS_S = (0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
QUANTITES_DB = (-38, -35, -32, -28, -24, -20, -16, -12, -9, -6)  # énergie réverbérée par rapport au son direct
GAIN_MIN_DB = 2.0  # amélioration moyenne minimale de la ressemblance pour appliquer une pièce
FINS_MIN = 4
BLOC_S = 10.0


def _bande(taux: int):
    return butter(4, BANDE_HZ, "bandpass", fs=taux, output="sos")


def energies(chemin: Path) -> tuple[np.ndarray, int]:
    """Énergie (dB) par trame de 5 ms dans la bande de la voix, lue par blocs (mémoire bornée)."""
    morceaux = []
    with sf.SoundFile(chemin) as f:
        taux = f.samplerate
        sos, etat = _bande(taux), None
        pas = max(1, int(TRAME_S * taux))
        reste = np.zeros(0, dtype=np.float32)
        for bloc in f.blocks(blocksize=int(BLOC_S * taux), dtype="float32", always_2d=True):
            x = bloc.mean(axis=1)
            if etat is None:
                etat = np.zeros((sos.shape[0], 2))
            y, etat = sosfilt(sos, x, zi=etat)
            y = np.concatenate([reste, y.astype(np.float32)])
            n = len(y) // pas
            morceaux.append(10 * np.log10(np.mean(y[: n * pas].reshape(n, pas) ** 2, axis=1) + 1e-12))
            reste = y[n * pas:]
    return (np.concatenate(morceaux) if morceaux else np.zeros(0)), taux


def fins_de_parole(e: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Trames où la parole s'arrête (suivies d'un vrai silence), et masque de parole."""
    if len(e) == 0:
        return [], np.zeros(0, dtype=bool)
    actif = e > np.percentile(e, 95) - 30
    apres = int(SILENCE_APRES_S / TRAME_S)
    fins = [i for i in range(1, len(actif) - apres) if actif[i - 1] and not actif[i] and not actif[i:i + apres].any()]
    return fins, actif


def courbe(e: np.ndarray, fins: list[int]) -> np.ndarray | None:
    """Courbe médiane de chute (dB sous la crête) sur DUREE_COURBE_S après chaque fin de parole."""
    n, avant = int(DUREE_COURBE_S / TRAME_S), int(AVANT_S / TRAME_S)
    chutes = [e[max(0, i - avant):i].max() - e[i:i + n] for i in fins if i + n <= len(e)]
    return np.median(np.array(chutes), axis=0) if chutes else None


def profil_origine(voix_isolee: Path) -> tuple[np.ndarray, float] | None:
    """Courbe de chute de la voix d'origine et plafond de mesure (dB) imposé par son bruit de fond."""
    e, _ = energies(voix_isolee)
    fins, actif = fins_de_parole(e)
    if len(fins) < FINS_MIN or actif.all():
        return None
    c = courbe(e, fins)
    if c is None:
        return None
    a, b, avant = int(BRUIT_S[0] / TRAME_S), int(BRUIT_S[1] / TRAME_S), int(AVANT_S / TRAME_S)
    bruit = [e[max(0, i - avant):i].max() - np.median(e[i + a:i + b]) for i in fins if i + b <= len(e)]
    plafond = float(np.median(bruit)) - 3.0
    return c, plafond


def reponse_impulsionnelle(reverberation_s: float, quantite_db: float, taux: int, graine: int = 7) -> np.ndarray:
    """Réponse d'une pièce simulée : son direct, puis réverbération (bruit qui s'éteint en `reverberation_s`
    de 60 dB), adoucie dans les aigus, d'énergie `quantite_db` par rapport au son direct."""
    n = int(min(1.5 * reverberation_s, 2.0) * taux)
    t = np.arange(n) / taux
    queue = np.random.default_rng(graine).standard_normal(n) * np.exp(-6.91 * t / reverberation_s)
    queue = sosfilt(butter(2, (150, min(5000, taux / 2 - 100)), "bandpass", fs=taux, output="sos"), queue)
    queue[: int(0.005 * taux)] = 0.0  # 5 ms entre le son direct et les premières réflexions
    queue *= np.sqrt(10 ** (quantite_db / 10) / max(float(np.sum(queue ** 2)), 1e-12))
    ri = queue.astype(np.float32)
    ri[0] = 1.0
    return ri / np.sqrt(1.0 + 10 ** (quantite_db / 10))  # volume d'ensemble inchangé


def _extraits(voix: Path, fins: list[int]) -> tuple[list[np.ndarray], int]:
    """Morceaux de la voix doublée autour de ses fins de parole (1,5 s avant, 0,5 s après)."""
    choisies = fins if len(fins) <= FINS_MAX else [fins[int(k)] for k in np.linspace(0, len(fins) - 1, FINS_MAX)]
    morceaux = []
    with sf.SoundFile(voix) as f:
        taux = f.samplerate
        for i in choisies:
            debut = max(0, int((i * TRAME_S - 1.5) * taux))
            f.seek(debut)
            morceaux.append((f.read(int(2.0 * taux), dtype="float32", always_2d=True).mean(axis=1), i * TRAME_S - debut / taux))
    return morceaux, taux


def _courbe_extraits(morceaux, taux: int, ri: np.ndarray | None) -> np.ndarray | None:
    pas, n, avant = max(1, int(TRAME_S * taux)), int(DUREE_COURBE_S / TRAME_S), int(AVANT_S / TRAME_S)
    sos, chutes = _bande(taux), []
    for x, fin_s in morceaux:
        y = fftconvolve(x, ri)[: len(x)] if ri is not None else x
        y = sosfilt(sos, y)
        m = len(y) // pas
        e = 10 * np.log10(np.mean(y[: m * pas].reshape(m, pas) ** 2, axis=1) + 1e-12)
        i = int(round(fin_s / TRAME_S))
        if avant <= i and i + n <= len(e):
            chutes.append(e[i - avant:i].max() - e[i:i + n])
    return np.median(np.array(chutes), axis=0) if chutes else None


def _ecart(c: np.ndarray, cible: np.ndarray, plafond: float) -> float:
    """Écart moyen (dB) entre deux courbes de chute, au-dessus du bruit de fond d'origine."""
    return float(np.mean(np.abs(np.minimum(c, plafond) - np.minimum(cible, plafond))))


def choisir_piece(voix_isolee: Path, voix_doublee: Path) -> dict | None:
    """Pièce simulée qui rapproche le plus la résonance de la voix doublée de celle d'origine, ou None."""
    profil = profil_origine(voix_isolee)
    if profil is None:
        return None
    cible, plafond = profil
    e, _ = energies(voix_doublee)
    fins, _ = fins_de_parole(e)
    if len(fins) < FINS_MIN:
        return None
    morceaux, taux = _extraits(voix_doublee, fins)
    seche = _courbe_extraits(morceaux, taux, None)
    if seche is None:
        return None
    avant = _ecart(seche, cible, plafond)
    meilleure = None
    for reverberation in REVERBERATIONS_S:
        for quantite in QUANTITES_DB:
            c = _courbe_extraits(morceaux, taux, reponse_impulsionnelle(reverberation, quantite, taux))
            if c is None:
                continue
            ecart = _ecart(c, cible, plafond)
            if meilleure is None or ecart < meilleure["ecart"] - 0.05:
                meilleure = {"reverberation_s": reverberation, "quantite_db": quantite, "ecart": ecart}
    if meilleure is None or avant - meilleure["ecart"] < GAIN_MIN_DB:
        logger.info("Acoustique : aucune pièce appliquée (écart %.1f dB, plafond de mesure %.0f dB)", avant, plafond)
        return None
    meilleure["ecart_avant"] = avant
    logger.info("Acoustique : réverbération %.2f s, %+d dB (écart de résonance %.1f → %.1f dB)",
                meilleure["reverberation_s"], meilleure["quantite_db"], avant, meilleure["ecart"])
    return meilleure


def appliquer_piece(entree: Path, sortie: Path, reverberation_s: float, quantite_db: float) -> Path:
    """Applique la pièce à toute la piste (convolution par blocs, avec report de la réverbération)."""
    with sf.SoundFile(entree) as f:
        taux, canaux = f.samplerate, f.channels
        ri = reponse_impulsionnelle(reverberation_s, quantite_db, taux)
        report = np.zeros((len(ri) - 1, canaux), dtype=np.float32)
        with sf.SoundFile(sortie, "w", samplerate=taux, channels=canaux, subtype="PCM_16") as s:
            for bloc in f.blocks(blocksize=int(BLOC_S * taux), dtype="float32", always_2d=True):
                y = np.stack([oaconvolve(bloc[:, c], ri) for c in range(canaux)], axis=1).astype(np.float32)
                y[: len(report)] += report  # réverbération du bloc précédent
                s.write(np.clip(y[: len(bloc)], -1.0, 1.0))
                report = y[len(bloc):]
    return sortie
