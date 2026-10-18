"""Tests du choix du moteur de voix (Chatterbox ou XTTS v2), sans modèle."""

import pytest

import config
from pipeline import voice, voice_bank


@pytest.fixture
def reglage(monkeypatch):
    def regler(moteur, installe):
        monkeypatch.setattr(config, "lire_parametres", lambda: {"moteur_voix": moteur})
        monkeypatch.setattr(voice, "chatterbox_disponible", lambda: installe)
    return regler


def test_choix_du_moteur(reglage):
    reglage("auto", True)
    assert voice.moteur_voix("fr") == "chatterbox"
    assert voice.moteur_voix("cs") == "xtts"  # tchèque : pas dans Chatterbox
    reglage("auto", False)
    assert voice.moteur_voix("fr") == "xtts"
    reglage("chatterbox", False)
    assert voice.moteur_voix("fr") == "xtts"  # pas encore installé
    reglage("xtts", True)
    assert voice.moteur_voix("fr") == "xtts"


def test_voix_de_bibliotheque_remplacee_par_son_apercu(tmp_path, monkeypatch):
    apercu = tmp_path / "daisy.wav"
    apercu.write_bytes(b"son")
    monkeypatch.setattr(voice_bank, "lire_banque", lambda: [{"id": "daisy", "xtts": "Daisy Studious"},
                                                            {"id": "perso", "source": "perso"}])
    monkeypatch.setattr(voice_bank, "fichier_apercu", lambda ident: tmp_path / f"{ident}.wav")
    converties = voice.voix_pour_chatterbox({1: "xtts:Daisy Studious|pitch=1.45", 2: "ref:/voix/franky.wav"})
    assert converties == {1: f"ref:{apercu}|pitch=1.45", 2: "ref:/voix/franky.wav"}
    # Une voix sans aperçu : tout le projet reste sur XTTS
    assert voice.voix_pour_chatterbox({1: "xtts:Inconnue"}) is None


def test_xtts_prend_le_relais_si_chatterbox_echoue(tmp_path, monkeypatch, reglage):
    reglage("auto", True)
    monkeypatch.setattr(voice, "voix_pour_chatterbox", lambda voix: voix)
    monkeypatch.setattr(voice, "_memoire_libre_go", lambda: 8.0)
    monkeypatch.setattr(voice, "vitesse_xtts", lambda *a: 1.0)
    lancements = []

    def lancer(commande, moteur, *args, **kwargs):
        lancements.append(moteur)
        if moteur == "chatterbox":
            raise voice.VoiceError("chatterbox cassé")
        return 1

    monkeypatch.setattr(voice, "_lancer_xtts", lancer)
    segments = [{"id": 1, "traduction": "Bonjour", "debut": 0.0, "fin": 1.0}]
    assert voice.generer_voix(segments, "fr", {1: "ref:/voix/a.wav"}, tmp_path) == 1
    assert lancements == ["chatterbox", "xtts"]
