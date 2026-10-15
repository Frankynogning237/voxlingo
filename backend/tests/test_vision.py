"""Tests de la lecture des scènes (parties sans modèle) et de son croisement avec l'écoute."""

from pipeline.diarize import PERSONNAGE, _avec_vision
from pipeline.vision import qui_parle, visibles

# 8 images/s ; personnage 0 de 0 à 2 s (bouche qui bouge), personnage 1 de 0 à 4 s (bouche immobile
# puis qui bouge de 2 à 4 s), personnage 2 seul à l'image de 6 à 8 s, bouche fermée
VISION = {
    "ips": 8,
    "personnages": 3,
    "pistes": [
        {"perso": 0, "images": list(range(0, 16)), "machoire": [0.0, 0.3] * 8},
        {"perso": 1, "images": list(range(0, 32)), "machoire": [0.1] * 16 + [0.0, 0.4] * 8},
        {"perso": 2, "images": list(range(48, 64)), "machoire": [0.05] * 16},
    ],
}


def test_qui_parle_designe_la_bouche_qui_bouge():
    assert qui_parle(VISION, 0.0, 2.0)[0] == 0
    assert qui_parle(VISION, 2.0, 4.0)[0] == 1
    assert qui_parle(VISION, 6.0, 8.0)[0] is None  # bouche fermée : l'image ne tranche pas
    assert qui_parle(VISION, 10.0, 11.0)[0] is None  # personne à l'image


def test_visibles():
    assert set(visibles(VISION, 6.5, 7.0)) == {2}
    assert set(visibles(VISION, 5.0, 5.5, marge=1.5)) == {1, 2}


def _mot(debut, fin, f0):
    return {"seg": 0, "mot": "x", "debut": debut, "fin": fin, "f0": f0}


def test_image_et_voix_croisees():
    mots = [_mot(0.0, 2.0, 120), _mot(2.0, 4.0, 300), _mot(6.0, 8.0, 125), _mot(10.0, 11.0, 300)]
    repliques = [[0], [1], [2], [3]]
    # Écoute : voix grave (groupe 0) pour les répliques 0 et 2, voix aiguë (groupe 1) pour 1 et 3
    groupes = _avec_vision(mots, repliques, [0, 1, 0, 1], VISION)
    assert groupes[0] == PERSONNAGE + 0  # bouche qui bouge
    assert groupes[1] == PERSONNAGE + 1
    # Personnage 2 à l'image, voix encore inconnue, bouche invisible : la voix grave lui revient
    assert groupes[2] == PERSONNAGE + 2
    # Hors champ : la voix aiguë, déjà vue chez le personnage 1
    assert groupes[3] == PERSONNAGE + 1
