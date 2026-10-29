"""Tests de l'identification des locuteurs (parties sans modèle : répliques, registres, découpage)."""

from pipeline.diarize import _decouper, _etiquette, _joindre, _repliques, _scinder_registres


def _mot(seg, mot, debut, fin, f0):
    return {"seg": seg, "mot": mot, "debut": debut, "fin": fin, "f0": f0}


def test_joindre_recolle_elisions_et_traits_d_union():
    assert _joindre(["Je", " vais", "d", "'un", "donne", "-moi", "ça", "."]) == "Je vais d'un donne-moi ça."


def test_repliques_coupees_aux_pauses_et_aux_sauts_de_hauteur():
    mots = [
        _mot(0, "a", 0.0, 0.3, 240), _mot(0, "b", 0.3, 0.6, 250), _mot(0, "c", 0.6, 0.9, 245),
        # même phrase, sans pause, mais la voix monte nettement (une autre personne répond)
        _mot(0, "d", 0.9, 1.2, 400), _mot(0, "e", 1.2, 1.5, 410), _mot(0, "f", 1.5, 1.8, 395),
        # pause d'une seconde
        _mot(0, "g", 2.8, 3.1, 390),
    ]
    assert _repliques(mots) == [[0, 1, 2], [3, 4, 5], [6]]


def test_repliques_coupees_aux_changements_pyannote():
    mots = [_mot(0, x, i * 0.3, i * 0.3 + 0.3, 200) for i, x in enumerate("abcd")]
    assert _repliques(mots, ["A", "A", "B", "B"]) == [[0, 1], [2, 3]]


def test_registres_distincts_scindes_mais_voix_continue_gardee():
    durees = [2.0] * 6
    # Deux personnages joués par la même voix de synthèse : ~240 Hz et ~400 Hz
    assert sorted(map(sorted, _scinder_registres(list(range(6)), [240, 250, 260, 380, 400, 420], durees))) == [
        [0, 1, 2], [3, 4, 5]]
    # Une vraie voix d'homme qui monte en s'énervant : pas de trou net, un seul locuteur
    assert len(_scinder_registres(list(range(6)), [110, 120, 130, 145, 160, 175], durees)) == 1


def test_etiquette_pyannote_par_recouvrement():
    tours = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]
    assert _etiquette(tours, 0.8, 1.6) == "B"
    assert _etiquette(tours, 2.5, 2.8) == "B"  # hors des tours : le plus proche


def test_phrase_decoupee_a_chaque_changement_de_locuteur():
    seg = {"id": 0, "debut": 0.0, "fin": 3.0, "texte": "Tu viens ? Non merci.", "ton": "neutre", "mots": [
        {"mot": "Tu", "debut": 0.0, "fin": 0.4}, {"mot": " viens", "debut": 0.4, "fin": 0.9},
        {"mot": " ?", "debut": 0.9, "fin": 1.0}, {"mot": " Non", "debut": 1.6, "fin": 2.1},
        {"mot": " merci.", "debut": 2.1, "fin": 3.0}]}
    mots = [_mot(0, m["mot"], m["debut"], m["fin"], None) for m in seg["mots"]]
    resultat = _decouper([seg], mots, ["S0", "S0", "S0", "S1", "S1"])
    assert [(s["id"], s["locuteur"], s["texte"], s["debut"], s["fin"]) for s in resultat] == [
        (0, "S0", "Tu viens ?", 0.0, 1.0), (1, "S1", "Non merci.", 1.6, 3.0)]
    assert resultat[1]["ton"] == "neutre"


def _replique_vue(debut: float, hauteur: float) -> list[dict]:
    return [{"debut": debut, "fin": debut + 1.0, "f0": hauteur}]


def _cas(lignes: list[tuple[float, int, int, float]], pistes: list[tuple[int, list[int]]]):
    """lignes : (début, voix pyannote, personnage vu, hauteur) ; pistes : (personnage, images à 8 i/s)."""
    from pipeline.diarize import PERSONNAGE

    mots, repliques = [], []
    for debut, _, _, hauteur in lignes:
        repliques.append([len(mots)])
        mots += _replique_vue(debut, hauteur)
    vision = {"ips": 8, "pistes": [{"perso": p, "images": images, "machoire": [0.0] * len(images)} for p, images in pistes]}
    return repliques, mots, [g for _, g, _, _ in lignes], [PERSONNAGE + p for _, _, p, _ in lignes], vision


def test_meme_voix_sous_plusieurs_visages_est_une_seule_personne():
    """Vidéo générée par IA : le père change de visage d'un plan à l'autre mais garde sa voix."""
    from pipeline.diarize import _regrouper

    lignes = [(0, 1, 0, 108), (10, 1, 1, 130), (20, 1, 10, 106), (30, 1, 2, 115), (40, 4, 8, 150)]
    pistes = [(0, [0, 8]), (1, [80, 88]), (10, [160]), (2, [240]), (8, [320])]
    resultat = _regrouper(*_cas(lignes, pistes))
    assert len(set(resultat[:4])) == 1  # le père : une seule personne
    assert resultat[4] != resultat[0]  # le fils, autre voix : distinct


def test_voix_partagee_par_des_personnages_vus_ensemble_garde_les_visages():
    """Dessin animé : une seule « voix » pour tous, mais des personnages vus ensemble à l'image."""
    from pipeline.diarize import _regrouper

    lignes = [(0, 0, 0, 349), (10, 0, 1, 370), (20, 0, 2, 335), (30, 0, 7, 248)]
    pistes = [(0, [0, 8, 240]), (7, [240]), (1, [80]), (2, [160])]
    assert len(set(_regrouper(*_cas(lignes, pistes)))) == 4


def test_un_visage_a_deux_voix_tres_differentes_reste_deux_personnes():
    """L'image confond le père et le fils (même visage) : leurs voix les séparent."""
    from pipeline.diarize import _regrouper

    lignes = [(0, 1, 0, 108), (10, 4, 0, 200)]
    resultat = _regrouper(*_cas(lignes, [(0, [0, 80])]))
    assert resultat[0] != resultat[1]
