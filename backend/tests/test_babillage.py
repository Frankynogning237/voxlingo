"""XTTS qui « babille » (sons ajoutés après la phrase) : la voix est refaite, jamais accélérée pour autant."""

import json
import sys
import types
from pathlib import Path

import numpy as np

from pipeline.align import duree_wav


def _faux_xtts(monkeypatch, durees: list[float]) -> list[tuple[float, float]]:
    """Faux XTTS : chaque appel rend une voix de la durée suivante de `durees`."""
    appels: list[tuple[float, float]] = []

    class Modele:
        def get_conditioning_latents(self, audio_path):
            return "latent", "empreinte"

        def inference(self, texte, langue, latent, empreinte, temperature, speed, enable_text_splitting):
            appels.append((temperature, speed))
            duree = durees[min(len(appels), len(durees)) - 1]
            return {"wav": np.full(int(duree * 24000), 0.2, dtype=np.float32)}

    class TTS:
        def __init__(self, *args, **kwargs):
            self.synthesizer = types.SimpleNamespace(tts_model=Modele())

        def to(self, device):
            return self

    monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))
    monkeypatch.setitem(sys.modules, "TTS.api", types.SimpleNamespace(TTS=TTS))
    return appels


def _generer(tmp_path: Path, disponible: float) -> Path:
    from pipeline import voice_worker

    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"")
    tache = tmp_path / "tache.json"
    tache.write_text(json.dumps({"langue": "de", "dossier": str(tmp_path), "segments": [
        {"id": 0, "traduction": "Noch dein Geschwätz-Zauber.", "vitesse": 1.0, "disponible": disponible,
         "voix": f"ref:{reference}"}]}), encoding="utf-8")
    voice_worker.main(str(tache))
    return tmp_path / "brut_00000.wav"


def test_voix_qui_babille_refaite_et_la_bonne_gardee(tmp_path, monkeypatch):
    appels = _faux_xtts(monkeypatch, [7.0, 7.0, 1.5])
    brut = _generer(tmp_path, disponible=2.0)
    assert [t for t, _ in appels] == [0.7, 0.5, 0.3]  # de plus en plus stable
    assert all(v == 1.0 for _, v in appels)  # jamais accélérée à cause du babillage
    assert duree_wav(brut) == 1.5


def test_voix_normale_generee_une_seule_fois(tmp_path, monkeypatch):
    appels = _faux_xtts(monkeypatch, [1.6])
    _generer(tmp_path, disponible=2.0)
    assert len(appels) == 1


def test_voix_qui_babille_toujours_garde_la_plus_courte(tmp_path, monkeypatch):
    appels = _faux_xtts(monkeypatch, [7.0, 5.0, 6.0])
    brut = _generer(tmp_path, disponible=2.0)
    assert len(appels) == 3 and all(v == 1.0 for _, v in appels)
    assert duree_wav(brut) == 5.0  # le calage coupera le surplus avant la réplique suivante
