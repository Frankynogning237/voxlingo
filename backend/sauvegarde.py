"""Sauvegarde automatique des données du moteur sur Hugging Face (Kaggle, Colab).

Kaggle et Colab effacent tous les fichiers à la fin de la séance (12 h max). Si un jeton
Hugging Face avec droit d'écriture est fourni (`HF_TOKEN_ECRITURE`), les projets terminés
(`data/projects/<id>`) et la bibliothèque de voix (`data/voix` : aperçus, voix personnelles)
sont recopiés dans un dépôt **privé** de type dataset :

- après chaque export terminé, et quand la bibliothèque de voix change ;
- à l'arrêt du moteur (tous les projets terminés).

Au démarrage d'une nouvelle séance, le carnet appelle `python -m sauvegarde restaurer`
pour tout remettre en place avant de lancer le moteur.

Le jeton n'est jamais écrit dans les journaux ni dans les messages.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from config import get_settings

logger = logging.getLogger("voxlingo.sauvegarde")

NOM_DEPOT_DEFAUT = "voxlingo-sauvegarde"
MARQUEUR_RESTAURATION = ".restauration-faite"
# Paquets Kaggle (recréés à la demande) et fichiers en cours d'écriture
IGNORES = ["*.zip", "*.tmp", "*.part", "**/*.tmp", "**/*.part"]


class Sauvegarde:
    """Synchronisation d'un dossier de données avec un dépôt dataset privé."""

    def __init__(self, jeton: str, data_dir: Path, depot: str | None = None) -> None:
        self._jeton = jeton
        self.data_dir = Path(data_dir)
        self._depot = depot or None
        self._depot_pret = False
        self._verrou = threading.Lock()  # une seule synchronisation à la fois
        self._verrou_attente = threading.Lock()
        self._attente: set[tuple[str, ...]] = set()
        self._fils: list[threading.Thread] = []

    # ── Outils ──────────────────────────────────────────────────────────────

    def masquer(self, texte: object) -> str:
        """Retire le jeton d'un message (erreurs de huggingface_hub comprises)."""
        return str(texte).replace(self._jeton, "***") if self._jeton else str(texte)

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(token=self._jeton)

    def depot(self) -> str:
        """Nom du dépôt (« compte/voxlingo-sauvegarde » par défaut), créé privé s'il n'existe pas."""
        api = self._api()
        if not self._depot:
            self._depot = f"{api.whoami()['name']}/{NOM_DEPOT_DEFAUT}"
        if not self._depot_pret:
            api.create_repo(self._depot, repo_type="dataset", private=True, exist_ok=True)
            self._depot_pret = True
        return self._depot

    def _envoyer(self, dossier: Path, chemin_depot: str, message: str) -> None:
        """Recopie `dossier` dans `chemin_depot` : les fichiers supprimés en local le sont aussi en ligne."""
        self._api().upload_folder(
            repo_id=self.depot(),
            repo_type="dataset",
            folder_path=str(dossier),
            path_in_repo=chemin_depot,
            commit_message=message,
            ignore_patterns=IGNORES,
            delete_patterns=["*"],
        )

    # ── Synchronisations (bloquantes) ───────────────────────────────────────

    def sauvegarder_projet(self, job_id: str) -> bool:
        dossier = self.data_dir / "projects" / job_id
        if not (dossier / "status.json").is_file():
            return False
        self._envoyer(dossier, f"projects/{job_id}", f"Projet {job_id}")
        logger.info("Sauvegarde du projet %s terminée.", job_id)
        return True

    def sauvegarder_voix(self) -> bool:
        dossier = self.data_dir / "voix"
        if not dossier.is_dir() or not any(dossier.iterdir()):
            return False
        self._envoyer(dossier, "voix", "Bibliothèque de voix")
        logger.info("Sauvegarde de la bibliothèque de voix terminée.")
        return True

    def supprimer_projet(self, job_id: str) -> None:
        try:
            self._api().delete_folder(f"projects/{job_id}", repo_id=self.depot(), repo_type="dataset",
                                      commit_message=f"Suppression du projet {job_id}")
        except Exception as exc:  # projet jamais sauvegardé : rien à supprimer
            if "404" not in str(exc) and "EntryNotFound" not in type(exc).__name__:
                raise

    def projets_termines(self) -> list[str]:
        import json

        ids = []
        for status in sorted((self.data_dir / "projects").glob("*/status.json")):
            try:
                if json.loads(status.read_text(encoding="utf-8")).get("etat") == "termine":
                    ids.append(status.parent.name)
            except ValueError:
                continue
        return ids

    def tout_sauvegarder(self) -> None:
        """Projets terminés et bibliothèque de voix (appelé à l'arrêt du moteur)."""
        for job_id in self.projets_termines():
            self._executer(("projet", job_id), lambda j=job_id: self.sauvegarder_projet(j))
        self._executer(("voix",), self.sauvegarder_voix)

    # ── File d'attente en arrière-plan ──────────────────────────────────────

    def _executer(self, cle: tuple[str, ...], action: Callable[[], object]) -> None:
        with self._verrou:
            with self._verrou_attente:
                self._attente.discard(cle)
            try:
                action()
            except Exception as exc:  # une sauvegarde ratée ne doit jamais casser le moteur
                logger.warning("Sauvegarde Hugging Face impossible (%s) : %s", " ".join(cle), self.masquer(exc))

    def planifier(self, cle: tuple[str, ...], action: Callable[[], object]) -> None:
        """Lance `action` en arrière-plan, sauf si la même sauvegarde attend déjà son tour."""
        with self._verrou_attente:
            if cle in self._attente:
                return
            self._attente.add(cle)
            fil = threading.Thread(target=self._executer, args=(cle, action), name="sauvegarde-hf", daemon=True)
            self._fils = [f for f in self._fils if f.is_alive()] + [fil]
        fil.start()

    def attendre(self, delai: float | None = None) -> None:
        """Attend la fin des sauvegardes lancées (tests, arrêt du moteur)."""
        for fil in list(self._fils):
            fil.join(delai)

    # ── Restauration (début de séance) ──────────────────────────────────────

    def restaurer(self) -> str:
        """Télécharge la dernière sauvegarde dans `data_dir`, une seule fois par séance."""
        marqueur = self.data_dir / MARQUEUR_RESTAURATION
        if marqueur.is_file():
            return "Données déjà restaurées pendant cette séance."
        from huggingface_hub import snapshot_download

        depot = self.depot()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=depot, repo_type="dataset", local_dir=str(self.data_dir), token=self._jeton)
        marqueur.write_text("ok", encoding="utf-8")
        projets = [p for p in (self.data_dir / "projects").glob("*") if (p / "status.json").is_file()]
        voix = self.data_dir / "voix"
        fichiers_voix = sum(1 for f in voix.rglob("*") if f.is_file()) if voix.is_dir() else 0
        return f"Restauré depuis {depot} : {len(projets)} projet(s), {fichiers_voix} fichier(s) de voix."


# ── Instance du moteur (configurée par .env ou par le carnet) ───────────────

_instance: Sauvegarde | None = None
_instance_lue = False


def instance() -> Sauvegarde | None:
    """La sauvegarde du moteur, ou None si aucun jeton d'écriture n'est fourni."""
    global _instance, _instance_lue
    if not _instance_lue:
        settings = get_settings()
        if settings.hf_token_ecriture:
            _instance = Sauvegarde(settings.hf_token_ecriture, settings.data_dir, settings.hf_depot_sauvegarde)
        _instance_lue = True
    return _instance


def apres_export(job_id: str) -> None:
    if sauvegarde := instance():
        sauvegarde.planifier(("projet", job_id), lambda: sauvegarde.sauvegarder_projet(job_id))


def apres_changement_voix() -> None:
    if sauvegarde := instance():
        sauvegarde.planifier(("voix",), sauvegarde.sauvegarder_voix)


def apres_suppression(job_id: str) -> None:
    if sauvegarde := instance():
        sauvegarde.planifier(("supprimer", job_id), lambda: sauvegarde.supprimer_projet(job_id))


def a_l_arret() -> None:
    if sauvegarde := instance():
        logger.info("Arrêt du moteur : sauvegarde des projets terminés et des voix…")
        sauvegarde.attendre(600)
        sauvegarde.tout_sauvegarder()


def main(arguments: list[str]) -> int:
    """`python -m sauvegarde restaurer` : utilisé par le carnet Kaggle/Colab au démarrage."""
    if arguments != ["restaurer"]:
        print("Usage : python -m sauvegarde restaurer")
        return 2
    sauvegarde = instance()
    if sauvegarde is None:
        print("Sauvegarde automatique désactivée (pas de jeton HF_TOKEN_ECRITURE).")
        return 0
    try:
        print(sauvegarde.restaurer())
    except Exception as exc:
        print("Restauration impossible :", sauvegarde.masquer(exc)[:500])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
