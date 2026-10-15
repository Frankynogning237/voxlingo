"""Identification des locuteurs : qui parle, et à quel moment de chaque phrase.

100 % local et sans compte. Trois étapes :

1. **Répliques** : la parole est découpée mot à mot en répliques, aux frontières des phrases,
   aux pauses et aux sauts nets de hauteur de voix (deux personnes qui se répondent sans pause).
2. **Empreintes vocales** : chaque réplique assez longue reçoit une empreinte (modèle public
   SpeechBrain ECAPA-TDNN, ~80 Mo) ; les empreintes proches sont regroupées (classification
   hiérarchique, distance cosinus). Le nombre de locuteurs est trouvé automatiquement.
3. **Registres** : un groupe dont les répliques se répartissent en deux hauteurs de voix bien
   distinctes est scindé. C'est le cas des vidéos générées par IA ou des doublages où une même
   voix de synthèse joue plusieurs personnages (un garçon à 240 Hz, une fille à 400 Hz…), que
   l'empreinte seule confond. Une vraie voix qui monte en s'énervant reste, elle, continue.

Les répliques trop courtes pour une empreinte prennent le locuteur voisin dont la voix a la
hauteur la plus proche. Enfin, avant la traduction, une phrase prononcée par plusieurs
personnes est découpée à chaque changement de locuteur.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.diarize")

TAUX = 16000
PAS_F0 = 160  # 10 ms
DUREE_MIN_EMPREINTE_S = 0.8  # en dessous, l'empreinte vocale n'est pas fiable
SEUIL_DISTANCE = 0.62  # distance cosinus moyenne au-delà de laquelle deux groupes sont des personnes différentes
PAUSE_REPLIQUE_S = 0.35  # silence qui sépare deux répliques
SAUT_HAUTEUR = 0.38  # saut de hauteur (en octave, ≈ +30 %) maintenu sur 2-3 mots : changement de personne
ECART_REGISTRES = 0.3  # trou de hauteur (octave) entre deux registres d'une même empreinte : deux personnages
PART_MIN_REGISTRE = 0.15  # chaque registre doit représenter au moins 15 % de la parole du groupe
DUREE_MIN_LOCUTEUR_S = 0.8  # un « locuteur » qui parle moins longtemps est rattaché au plus proche
DUREE_MIN_MORCEAU_S = 0.6  # morceau de phrase trop court pour en faire une phrase à part
DUREE_PETIT_LOCUTEUR_S = 2.5  # en dessous, un locuteur peut n'être qu'un écart de voix d'un autre
ECART_HAUTEUR_FUSION = 0.15  # même hauteur (octave)…
SEUIL_DISTANCE_FUSION = 0.8  # … et empreinte pas franchement différente : même personne
MODELE_PYANNOTE = "pyannote/speaker-diarization-community-1"
MARGE_VOISIN = 0.35  # une réplique courte suit son voisin sauf voix nettement plus proche d'un autre (octave)
COULEURS = ["#a855f7", "#3b82f6", "#22c55e", "#f97316", "#ec4899", "#eab308", "#14b8a6", "#ef4444"]


def _modele():
    if sys.platform == "win32":
        dll = Path(sys.prefix) / "Library" / "bin"
        if dll.is_dir():
            os.add_dll_directory(str(dll))
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(get_settings().models_dir / "speechbrain" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": get_runtime().device},
        # Windows refuse les liens symboliques sans droits administrateur : on copie les fichiers
        local_strategy=LocalStrategy.COPY,
    )


def _charger(source: Path, debut: float, fin: float) -> np.ndarray:
    """Extrait [debut, fin] en 16 kHz mono, sans charger tout le fichier (mémoire bornée)."""
    import librosa

    signal, _ = librosa.load(str(source), sr=TAUX, mono=True, offset=max(0.0, debut), duration=max(0.0, fin - debut))
    return signal


def _mots(segments: list[dict[str, Any]], source: Path,
          on_progress: Callable[[float], None] | None) -> list[dict[str, Any]]:
    """Mots de toutes les phrases (une phrase sans minutage des mots compte pour un seul « mot »),
    avec leur hauteur de voix médiane (`f0`, None si non voisé)."""
    import librosa

    mots: list[dict[str, Any]] = []
    for n, seg in enumerate(segments):
        liste = seg.get("mots") or [{"mot": seg["texte"], "debut": seg["debut"], "fin": seg["fin"]}]
        signal = _charger(source, seg["debut"], seg["fin"] + 0.1)
        f0 = voise = None
        if signal.size >= 2048:
            f0, voise, _ = librosa.pyin(signal, fmin=65, fmax=650, sr=TAUX, frame_length=1024, hop_length=PAS_F0)
        for mot in liste:
            hauteur = None
            if f0 is not None:
                a = max(0, int((mot["debut"] - seg["debut"]) * TAUX / PAS_F0))
                b = max(a + 1, int((max(mot["fin"], mot["debut"] + 0.12) - seg["debut"]) * TAUX / PAS_F0))
                valeurs = f0[a:b][voise[a:b] & ~np.isnan(f0[a:b])]
                hauteur = float(np.median(valeurs)) if valeurs.size >= 2 else None
            mots.append({"seg": n, "mot": mot["mot"], "debut": mot["debut"], "fin": mot["fin"], "f0": hauteur})
        if on_progress:
            on_progress((n + 1) / len(segments))
    return mots


def _mediane_f0(mots: list[dict[str, Any]]) -> float | None:
    valeurs = [m["f0"] for m in mots if m["f0"]]
    return float(np.median(valeurs)) if valeurs else None


def _repliques(mots: list[dict[str, Any]], etiquettes: list[str] | None = None) -> list[list[int]]:
    """Découpe la suite des mots en répliques (listes d'indices de mots), aussi à chaque changement
    de locuteur trouvé par pyannote (`etiquettes`, une par mot)."""
    repliques: list[list[int]] = []
    courante: list[int] = []
    for i, mot in enumerate(mots):
        if courante:
            prec = mots[courante[-1]]
            coupe = (mot["seg"] != prec["seg"] or mot["debut"] - prec["fin"] >= PAUSE_REPLIQUE_S
                     or (etiquettes is not None and etiquettes[i] != etiquettes[courante[-1]]))
            reference = _mediane_f0([mots[k] for k in courante[-4:]])
            suite = [m["f0"] for m in mots[i:i + 3] if m["f0"] and m["seg"] == mot["seg"]]
            if not coupe and reference and len(suite) >= 2:
                ecarts = [np.log2(f / reference) for f in suite]
                coupe = all(e > SAUT_HAUTEUR for e in ecarts) or all(e < -SAUT_HAUTEUR for e in ecarts)
            if coupe:
                repliques.append(courante)
                courante = []
        courante.append(i)
    if courante:
        repliques.append(courante)
    return repliques


def _scinder_registres(membres: list[int], hauteurs: list[float | None], durees: list[float]) -> list[list[int]]:
    """Scinde un groupe d'une même empreinte si ses répliques forment deux registres de hauteur
    séparés par un trou net (récursif : trois personnages ou plus sont possibles)."""
    mesures = sorted((np.log2(hauteurs[k]), k) for k in membres if hauteurs[k])
    total = sum(durees[k] for k in membres)
    meilleur, coupe = 0.0, None
    cumul = 0.0
    for j in range(len(mesures) - 1):
        cumul += durees[mesures[j][1]]
        ecart = mesures[j + 1][0] - mesures[j][0]
        part = cumul / total
        if (ecart >= ECART_REGISTRES and ecart > meilleur and PART_MIN_REGISTRE <= part <= 1 - PART_MIN_REGISTRE
                and min(cumul, total - cumul) >= DUREE_MIN_LOCUTEUR_S):
            meilleur, coupe = ecart, (mesures[j][0] + mesures[j + 1][0]) / 2
    if coupe is None:
        return [membres]
    bas = [k for k in membres if not hauteurs[k] or np.log2(hauteurs[k]) < coupe]
    haut = [k for k in membres if hauteurs[k] and np.log2(hauteurs[k]) >= coupe]
    return _scinder_registres(bas, hauteurs, durees) + _scinder_registres(haut, hauteurs, durees)


def _plus_proche(hauteur: float | None, candidats: list[int], hauteur_groupe: dict[int, float | None]) -> int:
    """Groupe dont la hauteur médiane est la plus proche (le premier candidat si rien n'est mesurable)."""
    mesurables = [g for g in candidats if hauteur and hauteur_groupe.get(g)]
    if not mesurables:
        return candidats[0]
    return min(mesurables, key=lambda g: abs(np.log2(hauteur / hauteur_groupe[g])))  # type: ignore[operator]


def _attribuer(mots: list[dict[str, Any]], repliques: list[list[int]], empreintes: dict[int, np.ndarray],
               etiquettes_pyannote: list[str] | None = None) -> list[int]:
    """Groupe (locuteur) de chaque réplique : d'après pyannote s'il a tourné, sinon d'après les empreintes."""
    from sklearn.cluster import AgglomerativeClustering

    hauteurs = [_mediane_f0([mots[k] for k in r]) for r in repliques]
    durees = [mots[r[-1]]["fin"] - mots[r[0]]["debut"] for r in repliques]
    if etiquettes_pyannote is not None:
        # 1. Locuteurs trouvés par pyannote (toutes les répliques en ont un)
        longues = list(range(len(repliques)))
        noms = {nom: n for n, nom in enumerate(dict.fromkeys(etiquettes_pyannote))}
        etiquettes = [noms[nom] for nom in etiquettes_pyannote]
    else:
        longues = sorted(empreintes)
        if not longues:
            return [0] * len(repliques)
        etiquettes = [0]
    # 1. Empreintes vocales (sans pyannote)
    if etiquettes_pyannote is None and len(longues) >= 2:
        etiquettes = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=SEUIL_DISTANCE
        ).fit_predict(np.stack([empreintes[i] for i in longues])).tolist()
    # 2. Registres de hauteur à l'intérieur d'une même empreinte
    groupe: dict[int, int] = {}
    numero = 0
    for etiquette in sorted(set(etiquettes)):
        membres = [i for i, e in zip(longues, etiquettes, strict=True) if e == etiquette]
        for partie in _scinder_registres(membres, hauteurs, durees):
            for i in partie:
                groupe[i] = numero
            numero += 1

    def hauteurs_groupes() -> dict[int, float | None]:
        return {g: _mediane_f0([mots[k] for i, gi in groupe.items() if gi == g for k in repliques[i]])
                for g in set(groupe.values())}

    def centre(g: int) -> np.ndarray:
        v = np.mean([empreintes[i] for i, gi in groupe.items() if gi == g], axis=0)
        return v / (np.linalg.norm(v) or 1.0)

    # 3. Locuteurs qui parlent à peine : une réplique isolée, souvent un simple écart de la voix
    #    d'un autre (cri, chuchotement). Rattachés au groupe de même hauteur et d'empreinte voisine,
    #    ou, s'ils parlent vraiment trop peu, au groupe de hauteur la plus proche.
    fusionnes: set[int] = set()
    while len(set(groupe.values())) > 1:
        parole = {g: sum(durees[i] for i, gi in groupe.items() if gi == g) for g in set(groupe.values())}
        h = hauteurs_groupes()
        petit = cible = None
        for g in sorted(parole, key=parole.__getitem__):
            if parole[g] >= DUREE_PETIT_LOCUTEUR_S or g in fusionnes:
                continue
            autres = [a for a in parole if a != g]
            proche = _plus_proche(h[g], autres, h)
            ecart_hauteur = abs(np.log2(h[g] / h[proche])) if h[g] and h[proche] else 1.0
            # Sans empreintes (pyannote les a déjà comparées) : seule la durée décide
            ecart_empreinte = float(1 - centre(g) @ centre(proche)) if empreintes else 1.0
            logger.debug("Petit locuteur %d : %.1f s, écart de hauteur %.2f, d'empreinte %.2f",
                         g, parole[g], ecart_hauteur, ecart_empreinte)
            if parole[g] < DUREE_MIN_LOCUTEUR_S or (
                ecart_hauteur <= ECART_HAUTEUR_FUSION and ecart_empreinte <= SEUIL_DISTANCE_FUSION
            ):
                petit, cible = g, proche
                break
            fusionnes.add(g)  # vrai locuteur discret (le cousin au téléphone) : on le garde
        if petit is None:
            break
        groupe = {i: (cible if g == petit else g) for i, g in groupe.items()}

    # 4. Répliques courtes : le voisin (précédent ou suivant) dont la voix est la plus proche
    h = hauteurs_groupes()
    resultat: list[int] = []
    for i in range(len(repliques)):
        if i in groupe:
            resultat.append(groupe[i])
            continue
        voisins = [groupe[j] for j in (max((j for j in groupe if j < i), default=None),
                                       min((j for j in groupe if j > i), default=None)) if j is not None]
        proche_global = _plus_proche(hauteurs[i], sorted(h), h)
        proche_voisin = _plus_proche(hauteurs[i], voisins, h)
        # Le voisin l'emporte sauf si un autre locuteur a une voix nettement plus proche
        if hauteurs[i] and h.get(proche_voisin) and h.get(proche_global) and (
            abs(np.log2(hauteurs[i] / h[proche_voisin])) - abs(np.log2(hauteurs[i] / h[proche_global])) > 0.2
        ):
            resultat.append(proche_global)
        else:
            resultat.append(proche_voisin)
    return resultat


def _joindre(mots: list[str]) -> str:
    """Recolle des mots Whisper (« d » + « 'un » → « d'un », « donne » + « -moi » → « donne-moi »)."""
    texte = ""
    for mot in mots:
        mot = mot.strip()
        if not mot:
            continue
        colle = texte.endswith(("'", "’", "-")) or mot.startswith(("'", "’", "-", ",", ".", "…"))
        texte = f"{texte}{mot}" if colle or not texte else f"{texte} {mot}"
    return texte


def _decouper(segments: list[dict[str, Any]], mots: list[dict[str, Any]], code_mot: list[str]) -> list[dict[str, Any]]:
    """Découpe chaque phrase à chaque changement de locuteur (morceaux trop courts recollés au voisin)."""
    resultat: list[dict[str, Any]] = []
    for n, seg in enumerate(segments):
        indices = [k for k, m in enumerate(mots) if m["seg"] == n]
        morceaux: list[list[int]] = []
        for k in indices:
            if morceaux and code_mot[morceaux[-1][-1]] == code_mot[k]:
                morceaux[-1].append(k)
            else:
                morceaux.append([k])
        # Un morceau très court (souvent une erreur de frontière) rejoint le morceau voisin
        fusion: list[list[int]] = []
        for morceau in morceaux:
            duree = mots[morceau[-1]]["fin"] - mots[morceau[0]]["debut"]
            if fusion and (duree < DUREE_MIN_MORCEAU_S or len(morceau) == 1):
                fusion[-1].extend(morceau)
            else:
                fusion.append(morceau)
        if len(fusion) > 1:
            premier = fusion[0]
            if mots[premier[-1]]["fin"] - mots[premier[0]]["debut"] < DUREE_MIN_MORCEAU_S or len(premier) == 1:
                fusion[1] = premier + fusion[1]
                fusion.pop(0)
        if len(fusion) <= 1 or not seg.get("mots"):
            seg["locuteur"] = _majoritaire([code_mot[k] for k in indices], [mots[k] for k in indices])
            resultat.append(seg)
            continue
        for j, morceau in enumerate(fusion):
            liste = [seg["mots"][k - indices[0]] for k in morceau]
            # Locuteur du morceau : celui qui y parle le plus longtemps
            resultat.append({
                **{c: v for c, v in seg.items() if c not in ("mots", "texte", "debut", "fin", "locuteur")},
                "debut": seg["debut"] if j == 0 else liste[0]["debut"],
                "fin": seg["fin"] if j == len(fusion) - 1 else liste[-1]["fin"],
                "texte": _joindre([m["mot"] for m in liste]),
                "mots": liste,
                "locuteur": _majoritaire([code_mot[k] for k in morceau], [mots[k] for k in morceau]),
            })
    for n, seg in enumerate(resultat):
        seg["id"] = n
    return resultat


def _majoritaire(codes: list[str], mots: list[dict[str, Any]]) -> str:
    parole: dict[str, float] = {}
    for code, mot in zip(codes, mots, strict=True):
        parole[code] = parole.get(code, 0.0) + max(0.05, mot["fin"] - mot["debut"])
    return max(parole, key=parole.__getitem__)


def _empreintes(audio: Path, mots: list[dict[str, Any]], repliques: list[list[int]],
                progression: Callable[[float], None]) -> dict[int, np.ndarray]:
    """Empreinte vocale ECAPA de chaque réplique assez longue (sans pyannote)."""
    import torch

    modele = _modele()
    empreintes: dict[int, np.ndarray] = {}
    try:
        for i, replique in enumerate(repliques):
            debut, fin = mots[replique[0]]["debut"], mots[replique[-1]]["fin"]
            if fin - debut >= DUREE_MIN_EMPREINTE_S:
                extrait = _charger(audio, debut, fin)
                if extrait.size >= TAUX * DUREE_MIN_EMPREINTE_S * 0.8:
                    with torch.no_grad():
                        vecteur = modele.encode_batch(torch.from_numpy(extrait).unsqueeze(0)).squeeze().cpu().numpy()
                    empreintes[i] = vecteur / (np.linalg.norm(vecteur) or 1.0)
            progression(0.45 + 0.5 * (i + 1) / len(repliques))
    finally:
        del modele
        gc.collect()

    return empreintes


def _pyannote(audio: Path) -> list[tuple[float, float, str]] | None:
    """Tours de parole (début, fin, locuteur) trouvés par pyannote, l'outil de référence open source.

    Nécessite un jeton Hugging Face (HF_TOKEN dans .env) et l'acceptation, une seule fois, des
    conditions du modèle gratuit « pyannote/speaker-diarization-community-1 » sur huggingface.co.
    Renvoie None si pyannote est indisponible : l'analyse locale par empreintes prend le relais.
    """
    jeton = get_settings().hf_token
    if not jeton:
        logger.info("Pas de jeton Hugging Face : identification des locuteurs sans pyannote")
        return None
    try:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(MODELE_PYANNOTE, token=jeton,
                                            cache_dir=get_settings().models_dir / "pyannote")
        if pipeline is None:
            raise RuntimeError("modèle inaccessible (conditions acceptées sur huggingface.co ?)")
        pipeline.to(torch.device(get_runtime().device))
        import librosa

        signal, _ = librosa.load(str(audio), sr=TAUX, mono=True)
        sortie = pipeline({"waveform": torch.from_numpy(signal).unsqueeze(0), "sample_rate": TAUX})
        # pyannote 4 : la version « exclusive » ne garde qu'un locuteur à la fois (pratique pour un doublage)
        annotation = getattr(sortie, "exclusive_speaker_diarization", None) or getattr(
            sortie, "speaker_diarization", sortie)
        tours = [(tour.start, tour.end, str(nom)) for tour, _, nom in annotation.itertracks(yield_label=True)]
        logger.info("pyannote : %d tours de parole, %d locuteurs", len(tours), len({t[2] for t in tours}))
        return tours or None
    except Exception as exc:  # noqa: BLE001 — l'analyse locale prend le relais
        logger.warning("pyannote indisponible (%s) : analyse locale par empreintes vocales", exc)
        return None
    finally:
        gc.collect()


def _etiquette(tours: list[tuple[float, float, str]], debut: float, fin: float) -> str:
    """Locuteur pyannote d'un mot : celui qui parle le plus pendant le mot, sinon le tour le plus proche."""
    recouvrement: dict[str, float] = {}
    for d, f, nom in tours:
        commun = min(f, fin) - max(d, debut)
        if commun > 0:
            recouvrement[nom] = recouvrement.get(nom, 0.0) + commun
    if recouvrement:
        return max(recouvrement, key=recouvrement.__getitem__)
    milieu = (debut + fin) / 2
    return min(tours, key=lambda t: min(abs(t[0] - milieu), abs(t[1] - milieu)))[2]


def identifier_locuteurs(
    segments: list[dict[str, Any]],
    audio: Path,
    on_progress: Callable[[float], None] | None = None,
    decouper: bool = True,
) -> dict[str, dict[str, str]]:
    """Ajoute `locuteur` (« SPEAKER_00 »…) à chaque phrase et renvoie la fiche des locuteurs.

    `audio` : de préférence la voix isolée (sans musique). Avec `decouper`, une phrase où
    plusieurs personnes parlent est remplacée, dans la liste, par une phrase par réplique.
    """
    if not segments:
        return {}
    progression = on_progress or (lambda _f: None)
    tours = _pyannote(audio)
    # Avec pyannote : 50 % pour lui, 45 % pour les hauteurs ; sans : 45 % hauteurs, 50 % empreintes
    depart = 0.5 if tours else 0.0
    progression(depart)
    mots = _mots(segments, audio, lambda f: progression(depart + 0.45 * f))
    if tours:
        etiquettes = [_etiquette(tours, m["debut"], m["fin"]) for m in mots]
        repliques = _repliques(mots, etiquettes)
        groupe_replique = _attribuer(mots, repliques, {}, [
            _majoritaire([etiquettes[k] for k in r], [mots[k] for k in r]) for r in repliques
        ])
    else:
        repliques = _repliques(mots)
        groupe_replique = _attribuer(mots, repliques, _empreintes(audio, mots, repliques, progression))
    groupe_mot = [0] * len(mots)
    for replique, g in zip(repliques, groupe_replique, strict=True):
        for k in replique:
            groupe_mot[k] = g

    # Numérotation dans l'ordre d'apparition : SPEAKER_00 parle en premier
    ordre: dict[int, str] = {}
    for g in groupe_mot:
        ordre.setdefault(g, f"SPEAKER_{len(ordre):02d}")
    code_mot = [ordre[g] for g in groupe_mot]

    if decouper:
        segments[:] = _decouper(segments, mots, code_mot)
    else:
        for n, seg in enumerate(segments):
            indices = [k for k, m in enumerate(mots) if m["seg"] == n]
            seg["locuteur"] = _majoritaire([code_mot[k] for k in indices], [mots[k] for k in indices])

    # Codes renumérotés sans trou, dans l'ordre d'apparition dans les phrases
    renumero: dict[str, str] = {}
    for seg in segments:
        renumero.setdefault(seg["locuteur"], f"SPEAKER_{len(renumero):02d}")
        seg["locuteur"] = renumero[seg["locuteur"]]
    locuteurs = {
        code: {"nom": f"Locuteur {k + 1}", "couleur": COULEURS[k % len(COULEURS)]}
        for k, code in enumerate(renumero.values())
    }
    logger.info("%d locuteur(s) identifié(s), %d répliques, %d phrases", len(locuteurs), len(repliques), len(segments))
    progression(1.0)
    return locuteurs
