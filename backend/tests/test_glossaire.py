"""Tests du glossaire et des consignes de traduction d'un projet."""

from pipeline.translate import _prompt, consignes_du_projet, formater_consignes, lire_glossaire

SEGMENTS = [{"id": 0, "texte": "Ma meuf veut voir ma baraque."}]


def test_lire_glossaire_accepte_plusieurs_separateurs():
    texte = "ma meuf = meine Freundin\nbaraque → Bude\nwesh -> hey\nmytho: Lüge\nligne sans séparateur\n = vide"
    assert lire_glossaire(texte) == [
        ("ma meuf", "meine Freundin"), ("baraque", "Bude"), ("wesh", "hey"), ("mytho", "Lüge"),
    ]


def test_formater_consignes():
    assert formater_consignes("", "  ") == ""
    bloc = formater_consignes("baraque = Bude", "Registre familier.")
    assert '"baraque" -> "Bude"' in bloc and "Registre familier." in bloc


def test_consignes_ajoutees_au_prompt_seulement_dans_le_bloc():
    with consignes_du_projet("baraque = Bude", "Tutoiement."):
        dans_le_bloc = _prompt(SEGMENTS, 0, "fr", "de", [])
    apres = _prompt(SEGMENTS, 0, "fr", "de", [])
    assert '"baraque" -> "Bude"' in dans_le_bloc and "Tutoiement." in dans_le_bloc
    assert "baraque\" ->" not in apres and "Tutoiement" not in apres


def test_qwen3_sans_reflexion():
    from pipeline.translate import options_modele

    assert options_modele("qwen3:14b") == {"think": False}
    assert options_modele("qwen2.5:14b") == {}


def test_modele_absent_telecharge(monkeypatch):
    from pipeline import translate

    appels = []

    class Reponse:
        def __init__(self, donnees=None):
            self.donnees = donnees

        def json(self):
            return self.donnees

        def raise_for_status(self):
            pass

    monkeypatch.setattr(translate.httpx, "get", lambda url, **_: Reponse({"models": [{"name": "qwen2.5:7b"}]}))
    monkeypatch.setattr(translate.httpx, "post", lambda url, json, **_: appels.append((url, json)) or Reponse())
    assert translate.ollama_disponible("qwen2.5:7b") and appels == []
    assert translate.ollama_disponible("qwen3:14b")
    assert appels[0][0].endswith("/api/pull") and appels[0][1]["model"] == "qwen3:14b"
