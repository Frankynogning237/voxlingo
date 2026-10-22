"""Durée des phrases face à leur scène : les phrases longues sont calées morceau par morceau sur les pauses
d'origine, les traductions trop courtes sont allongées ; une traduction corrigée à la main n'est pas retouchée."""

from types import SimpleNamespace

import pytest

from pipeline import adjust
from pipeline.segments import (
    besoin_ajustement,
    est_trop_court,
    est_trop_long,
    marquer_fin_max,
    syllabes_pour_duree,
)

SOURCE, CIBLE = "en", "fr"


def _texte(syllabes: int) -> str:
    """Texte français d'environ `syllabes` syllabes (« ba » = une syllabe)."""
    return " ".join(["bala"] * (syllabes // 2) + (["ba"] if syllabes % 2 else []))


def _mots(debut: float, groupes: list[tuple[int, float]]) -> list[dict]:
    """Mots d'origine : `groupes` = (nombre de mots, pause après le groupe) ; 0,4 s par mot."""
    mots, t = [], debut
    for nombre, pause in groupes:
        for k in range(nombre):
            mot = "word." if k == nombre - 1 and pause else "word"
            mots.append({"mot": mot, "debut": round(t, 2), "fin": round(t + 0.35, 2)})
            t += 0.4
        t += pause
    return mots


def _phrase(ident: int, mots: list[dict], traduction: str, **autres) -> dict:
    return {"id": ident, "debut": mots[0]["debut"], "fin": mots[-1]["fin"], "mots": mots,
            "texte": " ".join(m["mot"] for m in mots), "traduction": traduction, "traduction_ia": traduction,
            "locuteur": "SPEAKER_01", **autres}


@pytest.fixture
def ia(monkeypatch):
    """IA simulée : traduit chaque morceau à la bonne longueur, allonge ou raccourcit sur demande."""
    appels = {"allonger": 0, "traduire": 0}

    def traduire(client, modele, morceaux, k, source, cible, deja):
        appels["traduire"] += 1
        m = morceaux[k]
        return _texte(syllabes_pour_duree((m["fin"] - m["debut"]) * 0.95, cible))

    def allonger(client, modele, original, traduction, source, cible, syllabes_min, syllabes_max):
        appels["allonger"] += 1
        return _texte((syllabes_min + syllabes_max) // 2)

    monkeypatch.setattr(adjust, "_traduire_segment", traduire)
    monkeypatch.setattr(adjust, "allonger", allonger)
    monkeypatch.setattr(adjust, "raccourcir", lambda c, m, o, t, s, ci, n: _texte(n))
    monkeypatch.setattr(adjust, "traduction_suspecte", lambda *a: False)
    monkeypatch.setattr(adjust, "meme_sens", lambda *a: True)
    monkeypatch.setattr(adjust, "repartir", lambda *a: None)  # par défaut : traduction morceau par morceau
    monkeypatch.setattr(adjust, "ollama_disponible", lambda modele: True)
    monkeypatch.setattr(adjust, "_decharger_ollama", lambda modele: None)
    monkeypatch.setattr(adjust, "get_runtime", lambda: SimpleNamespace(ollama_model="qwen2.5:7b"))
    return appels


def test_phrase_longue_calee_sur_les_pauses_d_origine(ia):
    mots = _mots(1.0, [(6, 0.6), (5, 0.5), (7, 0.7), (6, 0)])  # ~13 s, trois pauses
    emotion = {"emotion": "colère", "intensite": "forte"}
    longue = _phrase(0, mots, _texte(syllabes_pour_duree(10.5, CIBLE)), ton="intense", emotion=emotion)
    suivante = _phrase(1, _mots(16.0, [(4, 0)]), _texte(syllabes_pour_duree(1.5, CIBLE)))
    data = {"segments": [longue, suivante], "duree_s": 20.0}
    assert besoin_ajustement(longue, CIBLE) == "caler"

    bilan = adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")

    morceaux = data["segments"][:-1]
    assert bilan["calees"] == 1 and len(morceaux) >= 3
    assert [s["id"] for s in data["segments"]] == list(range(len(data["segments"])))
    # Les morceaux couvrent la phrase d'origine et commencent après les pauses d'origine
    assert morceaux[0]["debut"] == longue["debut"] and morceaux[-1]["fin"] == longue["fin"]
    assert all(a["fin"] <= b["debut"] for a, b in zip(morceaux, morceaux[1:]))
    for m in morceaux:
        assert m["ajustement"] == "calee" and m["locuteur"] == "SPEAKER_01"
        assert m["ton"] == "intense" and m["emotion"] == emotion
        assert not m["trop_long"] and not est_trop_court(m, m["traduction"], CIBLE)
    assert data["segments"][-1]["texte"] == suivante["texte"]  # phrase courte et juste : intacte


def test_traduction_trop_courte_allongee(ia):
    mots = _mots(0.0, [(10, 0)])  # 3,95 s sans pause
    seg = _phrase(0, mots, _texte(syllabes_pour_duree(1.6, CIBLE)))
    data = {"segments": [seg], "duree_s": 6.0}
    assert besoin_ajustement(seg, CIBLE) == "allonger"

    bilan = adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")

    s = data["segments"][0]
    assert bilan["allongees"] == 1 and s["ajustement"] == "allongee"
    assert not est_trop_court(s, s["traduction"], CIBLE)
    assert not est_trop_long(s, s["traduction"], SOURCE, CIBLE, "reformuler")
    assert s["traduction_ia"] == seg["traduction_ia"]  # « Revenir à la traduction de l'IA » reste possible


def test_allongement_sans_amelioration_pas_reessaye(ia, monkeypatch):
    monkeypatch.setattr(adjust, "allonger", lambda c, m, o, traduction, *a: traduction)  # rien de mieux
    seg = _phrase(0, _mots(0.0, [(10, 0)]), _texte(syllabes_pour_duree(1.6, CIBLE)))
    texte = seg["traduction"]
    data = {"segments": [seg], "duree_s": 6.0}
    bilan = adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")
    s = data["segments"][0]
    assert bilan["allongees"] == 0 and s["traduction"] == texte and s["allongement_essaye"]
    assert besoin_ajustement(s, CIBLE) is None


def test_traduction_corrigee_a_la_main_respectee(ia):
    seg = _phrase(0, _mots(0.0, [(10, 0)]), _texte(syllabes_pour_duree(1.6, CIBLE)))
    seg["traduction"] = _texte(syllabes_pour_duree(1.5, CIBLE))  # corrigée par l'utilisateur
    data = {"segments": [seg], "duree_s": 6.0}
    assert not adjust.a_ajuster(data["segments"], SOURCE, CIBLE, "reformuler")
    adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")
    assert ia["allonger"] == 0
    # … sauf si l'on demande l'ajustement de cette phrase précisément
    adjust.ajuster_segments(data, SOURCE, CIBLE, ids={0}, strategie="reformuler")
    assert ia["allonger"] >= 1 and data["segments"][0]["ajustement"] == "allongee"


def test_phrase_juste_non_concernee():
    mots = _mots(0.0, [(8, 0)])
    seg = _phrase(0, mots, _texte(syllabes_pour_duree(3.0, CIBLE)))
    marquer_fin_max([seg], 5.0)
    assert besoin_ajustement(seg, CIBLE) is None
    # Très courte réplique : jamais allongée (« Oui. » reste « Oui. »)
    courte = _phrase(0, _mots(0.0, [(2, 0)]), "oui")
    assert besoin_ajustement(courte, CIBLE) is None


def test_long_silence_apres_un_seul_mot_marque():
    """« Il… » puis 3,8 s de silence avant la suite : la coupure se fait au silence, même pour un seul mot."""
    from pipeline.segments import decouper

    mots = _mots(29.6, [(1, 3.8), (20, 0)])
    morceaux = decouper(_phrase(0, mots, "x"))
    assert [len(m["mots"]) for m in morceaux][0] == 1
    assert morceaux[1]["debut"] == pytest.approx(mots[1]["debut"])


def test_calage_reprend_la_traduction_complete(ia, monkeypatch):
    """La traduction de la phrase entière est répartie entre les morceaux, sans retraduire de fragments."""
    parts = []

    def repartir(client, modele, parties, traduction, source, cible):
        parts.append(parties)
        # 0,4 s par mot d'origine : chaque morceau traduit remplit son moment
        return [f"partie {k} " + _texte(syllabes_pour_duree(len(p.split()) * 0.38, cible))
                for k, p in enumerate(parties)]

    monkeypatch.setattr(adjust, "repartir", repartir)
    mots = _mots(1.0, [(6, 0.6), (5, 0.5), (7, 0)])
    data = {"segments": [_phrase(0, mots, _texte(syllabes_pour_duree(8.0, CIBLE)))], "duree_s": 12.0}
    adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")
    assert ia["traduire"] == 0 and len(parts) == 1
    assert [s["traduction"].split()[1] for s in data["segments"]] == [str(k) for k in range(len(data["segments"]))]


def test_allongement_qui_change_le_sens_refuse(ia, monkeypatch):
    monkeypatch.setattr(adjust, "meme_sens", lambda *a: False)
    seg = _phrase(0, _mots(0.0, [(10, 0)]), _texte(syllabes_pour_duree(1.6, CIBLE)))
    texte = seg["traduction"]
    data = {"segments": [seg], "duree_s": 6.0}
    bilan = adjust.ajuster_segments(data, SOURCE, CIBLE, strategie="reformuler")
    assert bilan["allongees"] == 0 and data["segments"][0]["traduction"] == texte


def test_reessai_sur_demande_explicite(ia):
    seg = _phrase(0, _mots(0.0, [(10, 0)]), _texte(syllabes_pour_duree(1.6, CIBLE)), allongement_essaye=True)
    data = {"segments": [seg], "duree_s": 6.0}
    assert not adjust.a_ajuster(data["segments"], SOURCE, CIBLE, "reformuler")
    bilan = adjust.ajuster_segments(data, SOURCE, CIBLE, ids={0}, strategie="reformuler")
    assert bilan["allongees"] == 1


def test_calibrage_du_debit(tmp_path):
    from pipeline import calibrage
    from pipeline.segments import duree_voix_estimee

    texte = _texte(30)
    avant = duree_voix_estimee(texte, CIBLE)
    assert calibrage.apprendre(CIBLE, [1.3, 1.25, 1.35, 1.3, 5.0]) == pytest.approx(1.3)  # 5,0 : aberrant
    assert duree_voix_estimee(texte, CIBLE) == pytest.approx(avant * 1.3)
    assert calibrage.apprendre(CIBLE, [1.0]) is None  # trop peu de mesures : rien n'est appris
    assert calibrage.facteur("ja") == 1.0
