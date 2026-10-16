"""Tests de la sauvegarde sur Hugging Face, sans réseau (huggingface_hub simulé)."""

import json
import sys
import types
from pathlib import Path

import pytest

import sauvegarde
from sauvegarde import Sauvegarde

JETON = "hf_secretEcriture123"


class FauxHub:
    """Remplace le module huggingface_hub et note chaque appel."""

    def __init__(self, depot_distant=None, erreur=None):
        self.appels = []
        self.depot_distant = depot_distant or {}
        self.erreur = erreur
        hub = self

        class HfApi:
            def __init__(self, token=None):
                hub.appels.append(("HfApi", token))

            def whoami(self):
                return {"name": "franky"}

            def create_repo(self, repo_id, **options):
                hub.appels.append(("create_repo", repo_id, options))

            def upload_folder(self, **options):
                if hub.erreur:
                    raise hub.erreur
                hub.appels.append(("upload_folder", options))

            def delete_folder(self, chemin, **options):
                hub.appels.append(("delete_folder", chemin, options))

        def snapshot_download(repo_id, repo_type, local_dir, token):
            hub.appels.append(("snapshot_download", repo_id, repo_type, token))
            for chemin, contenu in hub.depot_distant.items():
                fichier = Path(local_dir) / chemin
                fichier.parent.mkdir(parents=True, exist_ok=True)
                fichier.write_text(contenu, encoding="utf-8")

        self.module = types.SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download)

    def nommes(self, nom):
        return [a for a in self.appels if a[0] == nom]


@pytest.fixture
def hub(monkeypatch):
    faux = FauxHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", faux.module)
    return faux


def _projet(data_dir, job_id, etat="termine"):
    dossier = data_dir / "projects" / job_id
    (dossier / "voix").mkdir(parents=True)
    (dossier / "status.json").write_text(json.dumps({"id": job_id, "etat": etat}), encoding="utf-8")
    (dossier / "segments.json").write_text("{}", encoding="utf-8")
    (dossier / "doublage.mp4").write_bytes(b"video")
    return dossier


def test_sauvegarde_projet_dans_un_depot_prive(tmp_path, hub):
    _projet(tmp_path, "abc123")
    assert Sauvegarde(JETON, tmp_path).sauvegarder_projet("abc123")

    (_, depot, options), = hub.nommes("create_repo")
    assert depot == "franky/voxlingo-sauvegarde"
    assert options == {"repo_type": "dataset", "private": True, "exist_ok": True}
    (_, envoi), = hub.nommes("upload_folder")
    assert envoi["repo_id"] == depot and envoi["repo_type"] == "dataset"
    assert envoi["folder_path"] == str(tmp_path / "projects" / "abc123")
    assert envoi["path_in_repo"] == "projects/abc123"
    assert "*.zip" in envoi["ignore_patterns"]
    assert envoi["delete_patterns"] == ["*"]  # miroir : un fichier effacé en local l'est aussi en ligne


def test_depot_choisi_et_cree_une_seule_fois(tmp_path, hub):
    _projet(tmp_path, "a")
    _projet(tmp_path, "b")
    sauve = Sauvegarde(JETON, tmp_path, depot="moi/mes-doublages")
    sauve.sauvegarder_projet("a")
    sauve.sauvegarder_projet("b")
    assert [a[1] for a in hub.nommes("create_repo")] == ["moi/mes-doublages"]
    assert [a[1]["repo_id"] for a in hub.nommes("upload_folder")] == ["moi/mes-doublages"] * 2


def test_rien_a_sauvegarder(tmp_path, hub):
    sauve = Sauvegarde(JETON, tmp_path)
    assert not sauve.sauvegarder_projet("inconnu")
    assert not sauve.sauvegarder_voix()
    assert hub.nommes("upload_folder") == []


def test_sauvegarde_voix(tmp_path, hub):
    (tmp_path / "voix" / "apercus").mkdir(parents=True)
    (tmp_path / "voix" / "banque.json").write_text("[]", encoding="utf-8")
    assert Sauvegarde(JETON, tmp_path).sauvegarder_voix()
    (_, envoi), = hub.nommes("upload_folder")
    assert envoi["path_in_repo"] == "voix" and envoi["folder_path"] == str(tmp_path / "voix")


def test_tout_sauvegarder_ne_prend_que_les_projets_termines(tmp_path, hub):
    _projet(tmp_path, "fini")
    _projet(tmp_path, "encours", etat="a_valider")
    (tmp_path / "voix").mkdir()
    (tmp_path / "voix" / "banque.json").write_text("[]", encoding="utf-8")
    Sauvegarde(JETON, tmp_path).tout_sauvegarder()
    assert [a[1]["path_in_repo"] for a in hub.nommes("upload_folder")] == ["projects/fini", "voix"]


def test_erreur_reseau_journalisee_sans_le_jeton(tmp_path, monkeypatch, caplog):
    faux = FauxHub(erreur=RuntimeError(f"401 refusé pour le jeton {JETON}"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", faux.module)
    _projet(tmp_path, "abc")
    sauve = Sauvegarde(JETON, tmp_path)
    sauve.planifier(("projet", "abc"), lambda: sauve.sauvegarder_projet("abc"))
    sauve.attendre(5)  # aucune exception ne remonte : le moteur continue
    assert "401 refusé" in caplog.text
    assert JETON not in caplog.text and "***" in caplog.text


def test_planifier_en_arriere_plan(tmp_path, hub):
    _projet(tmp_path, "abc")
    sauve = Sauvegarde(JETON, tmp_path)
    sauve.planifier(("projet", "abc"), lambda: sauve.sauvegarder_projet("abc"))
    sauve.attendre(5)
    assert len(hub.nommes("upload_folder")) == 1


def test_planifier_ignore_un_doublon_en_attente(tmp_path, hub):
    import threading

    sauve = Sauvegarde(JETON, tmp_path)
    libere, appels = threading.Event(), []
    sauve.planifier(("bloque",), lambda: libere.wait(5))  # occupe la file
    for _ in range(3):
        sauve.planifier(("voix",), lambda: appels.append(1))
    libere.set()
    sauve.attendre(5)
    assert appels == [1]


def test_supprimer_projet(tmp_path, hub):
    Sauvegarde(JETON, tmp_path).supprimer_projet("abc")
    (_, chemin, options), = hub.nommes("delete_folder")
    assert chemin == "projects/abc" and options["repo_type"] == "dataset"


def test_restaurer_une_seule_fois_par_seance(tmp_path, monkeypatch):
    faux = FauxHub(depot_distant={
        "projects/abc/status.json": '{"etat": "termine"}',
        "projects/abc/doublage.mp4": "video",
        "voix/banque.json": "[]",
        "voix/apercus/v1.wav": "son",
    })
    monkeypatch.setitem(sys.modules, "huggingface_hub", faux.module)
    sauve = Sauvegarde(JETON, tmp_path / "donnees")

    message = sauve.restaurer()
    assert "1 projet(s)" in message and "2 fichier(s) de voix" in message
    assert JETON not in message
    assert (tmp_path / "donnees" / "projects" / "abc" / "doublage.mp4").read_text() == "video"
    assert faux.nommes("snapshot_download") == [
        ("snapshot_download", "franky/voxlingo-sauvegarde", "dataset", JETON)]

    assert "déjà restaurées" in sauve.restaurer()
    assert len(faux.nommes("snapshot_download")) == 1


def test_sans_jeton_rien_ne_se_passe(monkeypatch, hub, capsys):
    monkeypatch.setattr(sauvegarde, "_instance", None)
    monkeypatch.setattr(sauvegarde, "_instance_lue", True)
    sauvegarde.apres_export("abc")
    sauvegarde.apres_changement_voix()
    sauvegarde.a_l_arret()
    assert sauvegarde.main(["restaurer"]) == 0
    assert "désactivée" in capsys.readouterr().out
    assert hub.appels == []


def test_commande_restaurer_masque_le_jeton(tmp_path, monkeypatch, capsys):
    faux = FauxHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", faux.module)
    sauve = Sauvegarde(JETON, tmp_path)
    monkeypatch.setattr(sauvegarde, "_instance", sauve)
    monkeypatch.setattr(sauvegarde, "_instance_lue", True)
    monkeypatch.setattr(sauve, "restaurer", lambda: (_ for _ in ()).throw(RuntimeError(f"échec {JETON}")))
    assert sauvegarde.main(["restaurer"]) == 1
    sortie = capsys.readouterr().out
    assert "Restauration impossible" in sortie and JETON not in sortie
