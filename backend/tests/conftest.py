"""Réglages communs aux tests."""

import pytest


@pytest.fixture(autouse=True)
def calibrage_isole(tmp_path, monkeypatch):
    """Les tests n'utilisent ni ne modifient le calibrage du débit des voix appris sur les vrais projets."""
    from pipeline import calibrage

    monkeypatch.setattr(calibrage, "_fichier", lambda: tmp_path / "calibrage_debit.json")
    monkeypatch.setattr(calibrage, "_cache", {"mtime": None, "valeurs": {}})
