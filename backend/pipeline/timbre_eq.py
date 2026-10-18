"""Timbre de la voix doublée rapproché de la voix d'origine, locuteur par locuteur (égalisation).

Le clonage reproduit la voix, mais la couleur du son diffère souvent : micro, pièce, voix plus
sourde ou plus brillante que l'original. On mesure le spectre moyen de la voix d'origine du
locuteur (piste « voix isolée ») et celui de ses voix générées, par tiers d'octave de 80 Hz à
10 kHz, puis on corrige l'écart avec un égaliseur doux (FFmpeg firequalizer) :

- la correction est centrée (moyenne nulle) : elle change la couleur, pas le volume, calé ensuite
  au décibel près (pipeline/niveau.py) ;
- elle est lissée et plafonnée (±8 dB) : on rapproche le timbre sans déformer la voix.

Seules les trames où la voix est présente comptent ; la lecture se fait par extraits (mémoire bornée).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

# Centres des bandes en tiers d'octave, de 80 Hz à 10 kHz (les voix générées s'arrêtent à 12 kHz)
BANDES_HZ = [80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
             3150, 4000, 5000, 6300, 8000, 10000]
CORRECTION_MAX_DB = 8.0
DUREE_MAX_S = 60.0  # durée de voix analysée par locuteur, de chaque côté
FENETRE = 2048


def spectre_bandes(signal: np.ndarray, taux: int) -> np.ndarray | None:
    """Spectre moyen (dB) par tiers d'octave, sur les trames où la voix est présente."""
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) < FENETRE * 4:
        return None
    pas = FENETRE // 2
    n = (len(signal) - FENETRE) // pas
    fenetre = np.hanning(FENETRE)
    trames = np.stack([signal[i * pas: i * pas + FENETRE] for i in range(n)]) * fenetre
    energie = np.mean(trames**2, axis=1)
    db = 10 * np.log10(np.maximum(energie, 1e-12))
    actives = trames[db >= db.max() - 35.0]
    if len(actives) < 4:
        return None
    puissance = np.mean(np.abs(np.fft.rfft(actives, axis=1)) ** 2, axis=0)
    frequences = np.fft.rfftfreq(FENETRE, 1 / taux)
    bandes = []
    for centre in BANDES_HZ:
        masque = (frequences >= centre / 2 ** (1 / 6)) & (frequences < centre * 2 ** (1 / 6))
        if not masque.any() or centre * 2 ** (1 / 6) > taux / 2:
            return None
        bandes.append(10 * np.log10(max(float(np.mean(puissance[masque])), 1e-20)))
    return np.array(bandes)


def spectre_extraits(extraits: Iterable[tuple[np.ndarray, int]]) -> np.ndarray | None:
    """Spectre moyen de plusieurs extraits (pondéré par leur durée), dans la limite de DUREE_MAX_S."""
    somme, poids_total = None, 0.0
    for signal, taux in extraits:
        spectre = spectre_bandes(signal, taux)
        if spectre is None:
            continue
        poids = len(signal) / taux
        lineaire = 10 ** (spectre / 10) * poids
        somme = lineaire if somme is None else somme + lineaire
        poids_total += poids
        if poids_total >= DUREE_MAX_S:
            break
    if somme is None:
        return None
    return 10 * np.log10(somme / poids_total)


def correction(origine: np.ndarray, generee: np.ndarray) -> np.ndarray:
    """Gains (dB) par bande qui amènent la voix générée vers le timbre d'origine : lissés, centrés, plafonnés."""
    ecart = origine - generee
    lisse = np.convolve(np.pad(ecart, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    centre = lisse - np.mean(lisse)
    return np.clip(centre, -CORRECTION_MAX_DB, CORRECTION_MAX_DB)


def filtre_ffmpeg(gains: np.ndarray | None) -> str | None:
    """Filtre FFmpeg firequalizer correspondant, ou None si la correction est négligeable (< 1 dB)."""
    if gains is None or float(np.max(np.abs(gains))) < 1.0:
        return None
    points = ";".join(f"entry({f},{g:.1f})" for f, g in zip(BANDES_HZ, gains))
    return f"firequalizer=gain_entry='{points}'"


def correction_locuteur(voix_isolee: Path, segments_origine: list[dict], fichiers_generes: list[Path]) -> np.ndarray | None:
    """Correction de timbre d'un locuteur, d'après ses phrases d'origine et ses voix générées."""
    import soundfile as sf

    from pipeline.niveau import lire_extrait

    if not voix_isolee.is_file():
        return None
    origine = spectre_extraits(
        lire_extrait(voix_isolee, s["debut"], s["fin"])
        for s in sorted(segments_origine, key=lambda s: s["fin"] - s["debut"], reverse=True)
    )
    generee = spectre_extraits(sf.read(str(f), dtype="float32") for f in fichiers_generes if f.is_file())
    if origine is None or generee is None:
        return None
    return correction(origine, generee)
