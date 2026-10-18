"""Volume de la voix doublée calé au décibel près sur celui de la phrase d'origine.

Pour chaque phrase, on mesure le niveau de la voix d'origine (piste « voix isolée », sans la musique)
et celui de la voix générée, puis on applique la différence. Le niveau est une moyenne d'énergie sur
les seules trames où la voix est présente (« niveau actif ») : les silences et les respirations ne
faussent pas la mesure, une phrase criée reste criée, une phrase chuchotée reste chuchotée, et
l'équilibre voix / musique de la vidéo d'origine est conservé.

La lecture se fait par extrait (début → fin de la phrase) : mémoire bornée, quelle que soit la durée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRAME_S = 0.03  # 30 ms
PLAGE_DB = 35.0  # trames plus faibles que (trame la plus forte − 35 dB) : silence, ignorées
PLANCHER_DB = -60.0  # en dessous : silence absolu
GAIN_MIN_DB, GAIN_MAX_DB = -30.0, 24.0  # au-delà, la mesure est suspecte (phrase quasi vide)


def niveau_actif_db(signal: np.ndarray, taux: int) -> float | None:
    """Niveau moyen (dBFS) des trames où la voix est présente ; None si l'extrait est silencieux."""
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    longueur = max(1, int(TRAME_S * taux))
    n = len(signal) // longueur
    if n == 0:
        return None
    trames = signal[: n * longueur].astype(np.float64).reshape(n, longueur)
    puissance = np.mean(trames**2, axis=1)
    db = 10 * np.log10(np.maximum(puissance, 1e-12))
    actives = db[(db >= db.max() - PLAGE_DB) & (db >= PLANCHER_DB)]
    if actives.size == 0:
        return None
    return round(float(10 * np.log10(np.mean(10 ** (actives / 10)))), 2)


def lire_extrait(fichier: Path, debut: float, fin: float) -> tuple[np.ndarray, int]:
    """Extrait [début, fin] d'un fichier audio, sans charger le reste."""
    import soundfile as sf

    with sf.SoundFile(str(fichier)) as f:
        taux = f.samplerate
        f.seek(min(int(max(debut, 0.0) * taux), f.frames))
        signal = f.read(max(0, int((fin - debut) * taux)), dtype="float32")
    return signal, taux


def niveau_fichier_db(fichier: Path) -> float | None:
    import soundfile as sf

    signal, taux = sf.read(str(fichier), dtype="float32")
    return niveau_actif_db(signal, taux)


def gain_pour_egaler(niveau_origine: float | None, niveau_genere: float | None) -> float | None:
    """Gain (dB) qui amène la voix générée au niveau de la voix d'origine ; None si non mesurable."""
    if niveau_origine is None or niveau_genere is None:
        return None
    return round(float(np.clip(niveau_origine - niveau_genere, GAIN_MIN_DB, GAIN_MAX_DB)), 2)
