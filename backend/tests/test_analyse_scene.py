"""Tests de l'analyse de la scène : émotions, manière de dire, visage, scènes, fiche et traduction en contexte."""

import json

import pytest

from pipeline.emotions import _fusionner, decrire, expression_du_visage, maniere_de_dire
from pipeline.scene import _fenetres, formater_fiche, scene_a
from pipeline.translate import _prompt, contexte_de_la_scene

REFERENCE = {"energie_db": -20.0, "variation_dt": 2.0, "debit": 4.0}


def test_maniere_de_dire_par_rapport_au_meme_locuteur():
    assert maniere_de_dire({"energie_db": -10.0, "debit": 5.5, "variation_dt": 3.0}, REFERENCE) == [
        "crie", "rapide", "voix animée"]
    assert maniere_de_dire({"energie_db": -32.0, "debit": 2.5, "variation_dt": 1.0}, REFERENCE) == [
        "chuchote", "lent", "monocorde"]
    assert maniere_de_dire({"energie_db": -20.0, "debit": 4.0, "variation_dt": 2.0}, REFERENCE) == []
    assert maniere_de_dire(None, REFERENCE) == []


def test_fusion_voix_maniere_visage():
    colere = _fusionner(("colère", 0.9), ["crie"], "fronce les sourcils")
    assert colere["emotion"] == "colère" and colere["intensite"] == "forte"
    # La voix ne tranche pas : le visage précise
    assert _fusionner(("colère", 0.3), [], "sourit")["emotion"] == "joie"
    assert _fusionner(None, [], None) == {"emotion": "neutre", "intensite": "légère", "confiance": 0.0,
                                          "maniere": [], "visage": None}
    assert decrire(colere) == "colère (forte), crie ; visage : fronce les sourcils"
    assert decrire(None) == ""


def test_expression_du_visage_du_personnage_qui_parle():
    # Formes : sourire, moue, froncement, sourcils hauts, yeux écarquillés
    vision = {"ips": 10, "pistes": [
        {"perso": 0, "images": [0, 1, 2, 3], "expressions": [[0.6, 0, 0, 0, 0]] * 4},
        {"perso": 1, "images": [0, 1, 2, 3], "expressions": [[0, 0, 0.5, 0, 0]] * 4},
    ]}
    assert expression_du_visage(vision, 0, 0.0, 0.3) == "sourit"
    assert expression_du_visage(vision, 1, 0.0, 0.3) == "fronce les sourcils"
    assert expression_du_visage(vision, 2, 0.0, 0.3) is None
    assert expression_du_visage(None, 0, 0.0, 0.3) is None


def test_fenetres_regroupent_les_plans_courts():
    assert _fenetres([1.0, 1.5, 6.0], 10.0) == [(0.0, 6.0), (6.0, 10.0)]
    assert len(_fenetres([0.5 * k for k in range(1, 400)], 200.0)) <= 40
    scenes = [{"debut": 0.0, "fin": 6.0, "description": "Un salon."}, {"debut": 6.0, "fin": 10.0, "description": "Une rue."}]
    assert scene_a(scenes, 3.0) == "Un salon." and scene_a(scenes, 7.0) == "Une rue." and scene_a(scenes, 20) is None


def test_prompt_de_traduction_dans_la_scene():
    segments = [
        {"id": 0, "debut": 0.0, "fin": 2.0, "texte": "Tu rigoles ?", "locuteur": "SPEAKER_00"},
        {"id": 1, "debut": 2.0, "fin": 4.0, "texte": "Rends-moi ça !", "locuteur": "SPEAKER_01",
         "emotion": {"emotion": "colère", "intensite": "forte", "maniere": ["crie"], "visage": None}},
    ]
    fiche = formater_fiche({"genre": "Dispute entre frères", "resume": "Ils se battent pour une manette.",
                            "registre": "familier", "ton": "tension comique",
                            "personnages": [{"locuteur": "Yanis", "description": "grand frère"}]})
    with contexte_de_la_scene(fiche, [{"debut": 0.0, "fin": 5.0, "description": "Deux garçons sur un canapé."}],
                              {"SPEAKER_00": "Yanis", "SPEAKER_01": "Sami"}):
        prompt = _prompt(segments, 1, "fr", "en", ["Are you kidding?"])
    assert "VIDEO CONTEXT SHEET" in prompt and "Dispute entre frères" in prompt and "Yanis : grand frère" in prompt
    assert "Speaker: Sami" in prompt and "colère (forte), crie" in prompt
    assert "On screen: Deux garçons sur un canapé." in prompt
    assert 'Yanis: "Tu rigoles ?"' in prompt
    # Hors du bloc : la traduction reste sans contexte de scène
    assert "VIDEO CONTEXT SHEET" not in _prompt(segments, 1, "fr", "en", ["Are you kidding?"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import jobs
    import main
    from config import get_settings

    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    monkeypatch.setattr(get_settings(), "cle_acces", None)
    jobs.project_dir("p1").mkdir(parents=True)
    jobs.create_job("p1", "video.mp4", "input.mp4", "en", {})
    jobs.update_status("p1", etat="a_valider")
    jobs.write_segments("p1", {"langue": "fr", "segments": [
        {"id": 0, "debut": 0.0, "fin": 2.0, "texte": "Salut", "emotion": {
            "emotion": "neutre", "intensite": "légère", "confiance": 0.4, "maniere": ["rapide"], "visage": None}}]})
    return TestClient(main.app)


def test_corriger_emotion_et_fiche(client):
    reponse = client.put("/jobs/p1/segments/0/emotion", json={"emotion": "joie", "intensite": "forte"})
    assert reponse.status_code == 200
    emotion = reponse.json()["segments"][0]["emotion"]
    assert emotion["emotion"] == "joie" and emotion["corrigee"] and emotion["maniere"] == ["rapide"]
    assert client.put("/jobs/p1/segments/0/emotion", json={"emotion": "inconnue"}).status_code == 422
    assert client.put("/jobs/p1/segments/9/emotion", json={"emotion": "joie"}).status_code == 404

    fiche = {"genre": "Sketch", "resume": "Deux amis.", "registre": "familier", "ton": "humour",
             "personnages": [{"locuteur": "Locuteur 1", "description": "ado"}]}
    reponse = client.put("/jobs/p1/fiche", json=fiche)
    assert reponse.status_code == 200 and reponse.json()["fiche"] == {**fiche, "corrigee": True}

    import jobs
    assert json.loads((jobs.project_dir("p1") / "segments.json").read_text(encoding="utf-8"))["fiche"]["corrigee"]
