"""Tests du choix RoFormer / Demucs pour la séparation voix / fond (sans modèle ni FFmpeg)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import separate


@pytest.fixture
def projet(tmp_path, monkeypatch):
    """FFmpeg simulé : l'audio d'origine est « extrait » dans audio_hq.wav."""

    class Resultat:
        returncode = 0

    vrai_run = separate.subprocess.run

    def faux_ffmpeg(commande, **options):
        if commande[0] != "ffmpeg":
            return vrai_run(commande, **options)
        Path(commande[-1]).write_bytes(b"audio")
        return Resultat()

    monkeypatch.setattr(separate.subprocess, "run", faux_ffmpeg)
    monkeypatch.setattr(separate, "_require", lambda nom: nom)
    monkeypatch.setattr(separate, "get_runtime", lambda: SimpleNamespace(device="cpu"))
    return tmp_path


def _faux_modele(lancements, reussit_roformer=True):
    def executer(commande, on_progress, env=None):
        module = commande[commande.index("-m") + 1]
        lancements.append(module)
        if module == "pipeline.roformer_worker":
            sortie = Path(commande[commande.index(module) + 2])
            if not reussit_roformer:
                return 1, ["CUDA out of memory\n"]
            sortie.mkdir(parents=True)
            (sortie / "vocals.wav").write_bytes(b"voix roformer")
            (sortie / "no_vocals.wav").write_bytes(b"fond roformer")
        else:
            stems = Path(commande[commande.index("-o") + 1]) / separate.MODELE
            stems.mkdir(parents=True)
            (stems / "vocals.wav").write_bytes(b"voix demucs")
            (stems / "no_vocals.wav").write_bytes(b"fond demucs")
        if on_progress:
            on_progress(1.0)
        return 0, []

    return executer


def test_roformer_en_priorite(projet, monkeypatch):
    lancements = []
    monkeypatch.setattr(separate, "roformer_disponible", lambda: True)
    monkeypatch.setattr(separate, "_executer", _faux_modele(lancements))
    voix, fond = separate.separer(projet / "video.mp4", projet)
    assert lancements == ["pipeline.roformer_worker"]
    assert voix.read_bytes() == b"voix roformer" and fond.read_bytes() == b"fond roformer"
    assert sorted(p.name for p in projet.iterdir()) == ["fond.wav", "voix_isolee.wav"]  # rien ne traîne


def test_demucs_si_roformer_echoue(projet, monkeypatch):
    lancements = []
    monkeypatch.setattr(separate, "roformer_disponible", lambda: True)
    monkeypatch.setattr(separate, "_executer", _faux_modele(lancements, reussit_roformer=False))
    voix, fond = separate.separer(projet / "video.mp4", projet)
    assert lancements == ["pipeline.roformer_worker", "pipeline.demucs_worker"]
    assert voix.read_bytes() == b"voix demucs" and fond.read_bytes() == b"fond demucs"
    assert sorted(p.name for p in projet.iterdir()) == ["fond.wav", "voix_isolee.wav"]


def test_demucs_si_roformer_absent(projet, monkeypatch):
    lancements = []
    monkeypatch.setattr(separate, "roformer_disponible", lambda: False)
    monkeypatch.setattr(separate, "_executer", _faux_modele(lancements))
    separate.separer(projet / "video.mp4", projet)
    assert lancements == ["pipeline.demucs_worker"]


def test_echec_des_deux_message_lisible(projet, monkeypatch):
    monkeypatch.setattr(separate, "roformer_disponible", lambda: False)
    monkeypatch.setattr(separate, "_executer", lambda *a, **k: (1, ["erreur\n"]))
    with pytest.raises(separate.SeparationError, match="Conserver le fond sonore"):
        separate.separer(projet / "video.mp4", projet)
    assert not (projet / "audio_hq.wav").exists()
