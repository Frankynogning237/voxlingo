"""Traduction contextuelle locale via Ollama (Qwen 2.5), repli sur NLLB-200.

Chaque segment est traduit avec le contexte de ses voisins (avant et après),
en visant une longueur proche de l'original pour faciliter le doublage.
"""

from __future__ import annotations

import gc
import json
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx

from config import get_runtime, get_settings

logger = logging.getLogger("voxlingo.translate")

# Noms anglais pour le prompt : les petits modèles suivent mieux des consignes en anglais
LANGUAGE_NAMES = {
    "fr": "French", "en": "English", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "pl": "Polish", "tr": "Turkish", "ru": "Russian", "nl": "Dutch",
    "cs": "Czech", "ar": "Arabic", "zh-cn": "Simplified Chinese", "zh": "Simplified Chinese",
    "ja": "Japanese", "hu": "Hungarian", "ko": "Korean", "hi": "Hindi",
}

# Codes NLLB-200 (FLORES-200)
CODES_NLLB = {
    "fr": "fra_Latn", "en": "eng_Latn", "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "pl": "pol_Latn", "tr": "tur_Latn", "ru": "rus_Cyrl", "nl": "nld_Latn",
    "cs": "ces_Latn", "ar": "arb_Arab", "zh-cn": "zho_Hans", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "hu": "hun_Latn", "ko": "kor_Hang", "hi": "hin_Deva",
}

CONTEXTE = 2  # nombre de segments voisins envoyés de chaque côté
TOLERANCE_LONGUEUR = 0.15  # objectif du prompt : durée à ±15 %
# Le seuil « trop long » (temps disponible, silence compris) est dans pipeline/segments.py

# Débit moyen de parole en syllabes par seconde (Pellegrino, Coupé & Marsico, 2011).
# Une phrase espagnole a plus de syllabes que son équivalent anglais mais se prononce
# plus vite : on compare donc des durées estimées, pas des nombres de syllabes.
DEBIT_SYLLABES = {
    "en": 6.19, "fr": 7.18, "es": 7.82, "it": 6.99, "de": 5.97, "ja": 7.84,
    "zh": 5.18, "zh-cn": 5.18, "pt": 7.0, "nl": 6.0, "pl": 6.5, "cs": 6.5, "ru": 6.5,
    "tr": 7.0, "hu": 6.5, "ko": 7.0, "ar": 6.5, "hi": 6.5,
}
DEBIT_PAR_DEFAUT = 6.5

_VOYELLES = re.compile(r"[aeiouyàâäéèêëîïôöùûüœæ]+", re.IGNORECASE)
_CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")


def compter_syllabes(texte: str) -> int:
    """Estimation simple du nombre de syllabes (groupes de voyelles), suffisante pour comparer des longueurs."""
    if re.search(r"[぀-ヿ一-鿿가-힯]", texte):
        return len(re.findall(r"[぀-ヿ一-鿿가-힯]", texte))
    return max(1, len(_VOYELLES.findall(texte)))


def debit(langue: str | None) -> float:
    return DEBIT_SYLLABES.get(langue or "", DEBIT_PAR_DEFAUT)


def ecart_duree(original: str, traduction: str, source: str | None, cible: str | None) -> float:
    """Écart relatif de durée de parole estimée de la traduction par rapport à l'original."""
    duree_source = compter_syllabes(original) / debit(source)
    duree_cible = compter_syllabes(traduction) / debit(cible)
    return duree_cible / duree_source - 1


def syllabes_cibles(original: str, source: str | None, cible: str | None) -> int:
    """Nombre de syllabes à viser dans la langue cible pour garder la même durée."""
    return max(1, round(compter_syllabes(original) * debit(cible) / debit(source)))


class TranslationError(RuntimeError):
    """Erreur de traduction avec un message lisible."""


# ───────────────────────────── Ollama ─────────────────────────────

def ollama_disponible(modele: str) -> bool:
    """Vrai si le serveur Ollama répond et que le modèle est téléchargé (téléchargé au besoin)."""
    url = get_settings().ollama_url
    try:
        tags = httpx.get(f"{url}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Ollama ne répond pas sur %s", url)
        return False
    noms = {m.get("name") for m in tags.get("models", [])}
    if modele in noms or f"{modele}:latest" in noms:
        return True
    # Modèle choisi dans les paramètres mais pas encore téléchargé (Qwen3 sur Kaggle…) : on le récupère
    logger.info("Modèle %s absent d'Ollama : téléchargement (quelques minutes)…", modele)
    try:
        reponse = httpx.post(f"{url}/api/pull", json={"model": modele, "stream": False}, timeout=1800)
        reponse.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Téléchargement de %s impossible (%s) : ollama pull %s", modele, exc, modele)
        return False
    logger.info("Modèle %s téléchargé", modele)
    return True


def options_modele(modele: str) -> dict[str, Any]:
    """Qwen3 « réfléchit » avant de répondre : inutile (et lent) pour traduire une phrase."""
    return {"think": False} if modele.startswith("qwen3") else {}


# ───────────────────────── Glossaire et consignes du projet ─────────────────────────

LONGUEUR_MAX_CONSIGNES = 3000  # caractères : au-delà, le contexte du petit modèle déborde
_CONSIGNES: ContextVar[str] = ContextVar("consignes_du_projet", default="")


def lire_glossaire(texte: str) -> list[tuple[str, str]]:
    """« terme = traduction » (ou « → », « -> », « : »), une entrée par ligne ; le reste est ignoré."""
    entrees = []
    for ligne in (texte or "").splitlines():
        for separateur in ("=", "→", "->", ":"):
            if separateur in ligne:
                terme, _, traduction = ligne.partition(separateur)
                if terme.strip() and traduction.strip():
                    entrees.append((terme.strip(), traduction.strip()))
                break
    return entrees


def formater_consignes(glossaire: str, consignes: str) -> str:
    """Bloc ajouté aux demandes de traduction : glossaire imposé et consignes de style."""
    lignes = []
    entrees = lire_glossaire(glossaire)
    if entrees:
        lignes.append("Glossary imposed by the user (always translate these terms exactly like this):")
        lignes += [f'- "{terme}" -> "{traduction}"' for terme, traduction in entrees]
    if consignes and consignes.strip():
        lignes.append("Style instructions from the user (follow them):")
        lignes.append(consignes.strip())
    return "\n".join(lignes)[:LONGUEUR_MAX_CONSIGNES]


@contextmanager
def consignes_du_projet(glossaire: str = "", consignes: str = ""):
    """Pendant le bloc, chaque traduction et chaque reformulation suit le glossaire et les consignes."""
    jeton = _CONSIGNES.set(formater_consignes(glossaire, consignes))
    try:
        yield
    finally:
        _CONSIGNES.reset(jeton)


def _avec_consignes(lignes: list[str]) -> list[str]:
    fiche = _SCENE.get().get("fiche") or ""
    blocs = [b for b in (f"VIDEO CONTEXT SHEET:\n{fiche}" if fiche else "", _CONSIGNES.get()) if b]
    return [*(x for b in blocs for x in (b, "")), *lignes]


SYSTEM_PROMPT = (
    "You are a professional dubbing adapter. You have watched and listened to the whole video: you are "
    "given a context sheet, what is happening on screen, who is speaking and with which emotion and "
    "delivery. You adapt ONE line of the transcript at a time, as a native actor would really say it "
    "in this scene. Rules: (1) convey the meaning, the intention and the emotion of the line in this "
    "situation, never a word-for-word translation; (2) keep the speaker's register and tone (slang, "
    "casual, formal, child, teasing, angry…); (3) adapt idioms, jokes, insults and slang to natural "
    "equivalents in the target language; (4) keep the length close to the original so the dubbed voice "
    "fits the same time; (5) translate ONLY the line to translate, never the context lines; (6) write "
    "entirely in the target language, with correct spelling and grammar. "
    'Answer with strict JSON only: {"translation": "..."}'
)


# ───────────────────────── Contexte de la scène (analyse avant traduction) ─────────────────────────

_SCENE: ContextVar[dict[str, Any]] = ContextVar("contexte_de_la_scene", default={})


def genre_selon_voix(f0_hz: float | None) -> str | None:
    """« a man » / « a woman » d'après la hauteur de voix du locuteur, seulement quand elle est nette
    (entre 150 et 185 Hz, ou au-dessus de 260 Hz — enfants — on ne devine pas)."""
    if not f0_hz:
        return None
    if f0_hz < 150:
        return "a man"
    if 185 < f0_hz < 260:
        return "a woman"
    return None


@contextmanager
def contexte_de_la_scene(fiche: str = "", scenes: list[dict[str, Any]] | None = None,
                         noms: dict[str, str] | None = None, genres: dict[str, str | None] | None = None):
    """Pendant le bloc, chaque traduction reçoit la fiche de la vidéo, la scène en cours, le nom du
    locuteur, son genre (accords grammaticaux) et son émotion (voir pipeline/scene.py et pipeline/emotions.py)."""
    jeton = _SCENE.set({"fiche": fiche, "scenes": scenes or [], "noms": noms or {}, "genres": genres or {}})
    try:
        yield
    finally:
        _SCENE.reset(jeton)


def _nom(seg: dict[str, Any]) -> str:
    noms = _SCENE.get().get("noms") or {}
    return noms.get(seg.get("locuteur") or "", "")


def _lignes_de_scene(seg: dict[str, Any]) -> list[str]:
    """Qui parle, avec quelle émotion, et ce qui se passe à l'image, pour la ligne à traduire."""
    from pipeline.emotions import decrire
    from pipeline.scene import scene_a

    contexte = _SCENE.get()
    lignes = []
    genre = (contexte.get("genres") or {}).get(seg.get("locuteur") or "")
    if _nom(seg) or genre:
        qui = f"Speaker: {_nom(seg) or 'unnamed'}" + (f" ({genre})" if genre else "")
        lignes.append(qui)
    if genre:
        lignes.append(f"The speaker is {genre}: use the matching grammatical gender for every word referring "
                      "to the speaker (adjectives, past participles, self-references).")
    if seg.get("emotion"):
        lignes.append(f"Emotion and delivery (from the voice and the face): {decrire(seg['emotion'])}")
    scenes = contexte.get("scenes") or []
    scene = scene_a(scenes, (seg["debut"] + seg["fin"]) / 2) if scenes and "debut" in seg else None
    if scene:
        lignes.append(f"On screen: {scene}")
    return lignes


def _prompt(segments: list[dict[str, Any]], i: int, source: str, cible: str,
            deja_traduits: list[str], avec_contexte: bool = True) -> str:
    seg = segments[i]
    syllabes = syllabes_cibles(seg["texte"], source, cible)
    lignes = [
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
    ]
    if avec_contexte:
        def qui(j: int) -> str:
            return f"{_nom(segments[j])}: " if _nom(segments[j]) else ""

        avant = [
            f'- {qui(j)}"{segments[j]["texte"]}" -> "{deja_traduits[j]}"'
            for j in range(max(0, i - CONTEXTE), i)
        ]
        apres = [f'- {qui(j)}"{segments[j]["texte"]}"' for j in range(i + 1, min(len(segments), i + 1 + CONTEXTE))]
        if avant:
            lignes += ["", "Previous lines (already translated, for context only):", *avant]
        if apres:
            lignes += ["", "Next lines (for context only, do not translate):", *apres]
    lignes += ["", *_lignes_de_scene(seg)]
    lignes += [
        f'LINE TO TRANSLATE: "{seg["texte"]}"',
        f"Target length: about {syllabes} syllables "
        f"(between {max(1, round(syllabes * 0.85))} and {round(syllabes * 1.15)}).",
    ]
    return "\n".join(_avec_consignes(lignes))


def _normaliser(texte: str) -> str:
    return re.sub(r"[^\w]+", " ", texte.lower()).strip()


def traduction_suspecte(original: str, traduction: str, source: str, cible: str) -> bool:
    """Traduction probablement ratée : identique à l'original, mauvais alphabet, ou durée aberrante."""
    if _normaliser(original) == _normaliser(traduction) and len(_normaliser(original)) > 3:
        return True
    # Les petits modèles Qwen glissent parfois des caractères chinois (« meine真实的Hütte »)
    if cible not in ("zh", "zh-cn", "ja", "ko") and _CJK.search(traduction):
        return True
    ecart = ecart_duree(original, traduction, source, cible)
    return ecart < -0.5 or ecart > 1.0


def _traduire_ollama(client: httpx.Client, modele: str, prompt: str, temperature: float = 0.3) -> str:
    reponse = client.post(
        "/api/chat",
        json={
            "model": modele,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": {
                "type": "object",
                "properties": {"translation": {"type": "string"}},
                "required": ["translation"],
            },
            "stream": False,
            "options": {"temperature": temperature},
            "keep_alive": "2m",
            **options_modele(modele),
        },
    )
    reponse.raise_for_status()
    contenu = reponse.json()["message"]["content"]
    try:
        texte = json.loads(contenu)["translation"].strip().strip('"').strip()
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise TranslationError(f"Réponse inattendue du modèle : {contenu[:200]}") from exc
    if not texte:
        raise TranslationError("Le modèle a renvoyé une traduction vide.")
    return texte


def _traduire_segment(client: httpx.Client, modele: str, segments: list[dict[str, Any]], i: int,
                      source: str, cible: str, deja_traduits: list[str]) -> str:
    """Traduit un segment ; en cas de résultat suspect, réessaie sans contexte puis en mode strict."""
    original = segments[i]["texte"]
    texte = _traduire_ollama(client, modele, _prompt(segments, i, source, cible, deja_traduits))
    if not traduction_suspecte(original, texte, source, cible):
        return texte
    logger.info("Traduction suspecte du segment %d (%r), nouvel essai", i, texte)
    for avec_contexte, temperature in ((False, 0.0), (True, 0.7)):
        essai = _traduire_ollama(
            client, modele, _prompt(segments, i, source, cible, deja_traduits, avec_contexte), temperature
        )
        if not traduction_suspecte(original, essai, source, cible):
            return essai
    return texte


REVIEW_PROMPT = (
    "You are the dubbing editor. The whole dialogue has been translated line by line; you now review ONE "
    "line in the context of the lines around it. Fix it only if needed: wrong meaning, grammar or spelling "
    "errors, wrong grammatical gender for the speaker, inconsistent names, terms or forms of address "
    "(tu/vous, du/Sie…) with the other lines, or phrasing a native speaker would not say in this scene. "
    "Keep the length close to the current translation. If the line is already correct and natural, return "
    "it unchanged. Answer with strict JSON only: {\"translation\": \"...\"}"
)
FENETRE_RELECTURE = 4  # lignes voisines montrées de chaque côté pendant la relecture


def _prompt_relecture(segments: list[dict[str, Any]], traductions: list[str], i: int,
                      source: str, cible: str) -> str:
    def ligne(j: int) -> str:
        qui = f"{_nom(segments[j])}: " if _nom(segments[j]) else ""
        return f'- {qui}"{segments[j]["texte"]}" -> "{traductions[j]}"'

    debut, fin = max(0, i - FENETRE_RELECTURE), min(len(segments), i + 1 + FENETRE_RELECTURE)
    lignes = [
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
        "", "Dialogue around the line (original -> current translation):",
        *(ligne(j) for j in range(debut, i)),
        f">>> {ligne(i)[2:]}",
        *(ligne(j) for j in range(i + 1, fin)),
        "", *_lignes_de_scene(segments[i]),
        f'LINE TO REVIEW: "{segments[i]["texte"]}"',
        f'Current translation: "{traductions[i]}"',
    ]
    return "\n".join(_avec_consignes(lignes))


def relire(client: httpx.Client, modele: str, segments: list[dict[str, Any]], traductions: list[str], i: int,
           source: str, cible: str) -> str:
    """Relecture d'une ligne dans son dialogue. La correction n'est gardée que si elle est plausible
    (pas suspecte, longueur proche) : sinon la traduction actuelle reste."""
    actuelle = traductions[i]
    try:
        reponse = client.post("/api/chat", json={
            "model": modele, "stream": False, "keep_alive": "2m", **options_modele(modele),
            "options": {"temperature": 0.2},
            "format": {"type": "object", "properties": {"translation": {"type": "string"}}, "required": ["translation"]},
            "messages": [{"role": "system", "content": REVIEW_PROMPT},
                         {"role": "user", "content": _prompt_relecture(segments, traductions, i, source, cible)}],
        })
        reponse.raise_for_status()
        corrigee = json.loads(reponse.json()["message"]["content"])["translation"].strip().strip('"').strip()
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
        logger.warning("Relecture de la ligne %d impossible (%s)", i, exc)
        return actuelle
    if not corrigee or corrigee == actuelle or traduction_suspecte(segments[i]["texte"], corrigee, source, cible):
        return actuelle
    rapport = compter_syllabes(corrigee) / max(compter_syllabes(actuelle), 1)
    return corrigee if 0.7 <= rapport <= 1.25 else actuelle


SHORTEN_PROMPT = (
    "You shorten dubbing translations so the dubbed voice fits the original timing. "
    "Rewrite the translation more concisely in the same target language: keep the meaning, "
    "the tone and a natural spoken style, drop filler words, prefer shorter synonyms. "
    'Answer with strict JSON only: {"translation": "..."}'
)


LENGTHEN_PROMPT = (
    "You lengthen dubbing translations that are too short for the original timing, so the dubbed voice "
    "does not stop while the character is still speaking. Rewrite the translation in the same target "
    "language with fuller, natural spoken phrasing: complete forms instead of contractions, natural "
    "connectors and light rephrasing. Keep exactly the same meaning and tone, and never add new "
    "information, names or facts that are not in the original line. "
    'Answer with strict JSON only: {"translation": "..."}'
)


def raccourcir(client: httpx.Client, modele: str, original: str, traduction: str,
               source: str, cible: str, syllabes_max: int) -> str:
    """Demande au modèle une version plus courte de la traduction (au plus `syllabes_max` syllabes)."""
    return _reformuler(client, modele, SHORTEN_PROMPT, [
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
        f'Original line: "{original}"',
        f'Current translation (too long): "{traduction}"',
        f"Maximum length: {syllabes_max} syllables.",
    ], traduction, "le raccourcissement")


def allonger(client: httpx.Client, modele: str, original: str, traduction: str,
             source: str, cible: str, syllabes_min: int, syllabes_max: int) -> str:
    """Demande au modèle une version un peu plus développée de la traduction, de même sens
    (entre `syllabes_min` et `syllabes_max` syllabes)."""
    return _reformuler(client, modele, LENGTHEN_PROMPT, [
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
        f'Original line: "{original}"',
        f'Current translation (too short for the timing): "{traduction}"',
        f"Target length: between {syllabes_min} and {syllabes_max} syllables.",
    ], traduction, "l'allongement")


def _reformuler(client: httpx.Client, modele: str, systeme: str, lignes: list[str], traduction: str,
                quoi: str) -> str:
    prompt = "\n".join(_avec_consignes(lignes))
    reponse = client.post(
        "/api/chat",
        json={
            "model": modele,
            "messages": [
                {"role": "system", "content": systeme},
                {"role": "user", "content": prompt},
            ],
            "format": {
                "type": "object",
                "properties": {"translation": {"type": "string"}},
                "required": ["translation"],
            },
            "stream": False,
            "options": {"temperature": 0.4},
            "keep_alive": "2m",
            **options_modele(modele),
        },
    )
    reponse.raise_for_status()
    try:
        texte = json.loads(reponse.json()["message"]["content"])["translation"].strip().strip('"').strip()
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise TranslationError(f"Réponse inattendue du modèle pendant {quoi}.") from exc
    return texte or traduction


def _decharger_ollama(modele: str) -> None:
    """Libère la mémoire occupée par le modèle Ollama avant les étapes suivantes."""
    try:
        httpx.post(f"{get_settings().ollama_url}/api/generate",
                   json={"model": modele, "keep_alive": 0}, timeout=10)
    except httpx.HTTPError:
        pass


# ───────────────────────────── NLLB (secours) ─────────────────────────────

def _traduire_nllb(segments: list[dict[str, Any]], source: str, cible: str,
                   on_progress: Callable[[float], None] | None) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if source not in CODES_NLLB or cible not in CODES_NLLB:
        raise TranslationError(f"Langue non prise en charge par NLLB : {source} → {cible}.")
    nom = get_settings().nllb_model
    cache = str(get_settings().models_dir / "nllb")
    runtime = get_runtime()
    logger.info("Chargement de %s (%s)", nom, runtime.device)
    tokenizer = AutoTokenizer.from_pretrained(nom, src_lang=CODES_NLLB[source], cache_dir=cache)
    model = AutoModelForSeq2SeqLM.from_pretrained(nom, cache_dir=cache).to(runtime.device)
    try:
        cible_id = tokenizer.convert_tokens_to_ids(CODES_NLLB[cible])
        resultats = []
        for i, seg in enumerate(segments):
            entree = tokenizer(seg["texte"], return_tensors="pt").to(runtime.device)
            sortie = model.generate(**entree, forced_bos_token_id=cible_id, max_new_tokens=256)
            resultats.append(tokenizer.batch_decode(sortie, skip_special_tokens=True)[0].strip())
            if on_progress:
                on_progress((i + 1) / len(segments))
        return resultats
    finally:
        del model
        gc.collect()
        if runtime.device == "cuda":
            import torch

            torch.cuda.empty_cache()


# ───────────────────────────── Point d'entrée ─────────────────────────────

def traduire_selection(
    segments: list[dict[str, Any]],
    ids: set[int],
    source: str,
    cible: str,
    on_progress: Callable[[float], None] | None = None,
) -> dict[int, str]:
    """Retraduit seulement les segments `ids` (texte d'origine corrigé), avec pour contexte
    les phrases voisines et leurs traductions actuelles. Renvoie {id: nouvelle traduction}."""
    indices = [i for i, s in enumerate(segments) if s["id"] in ids]
    if not indices:
        return {}
    if source == cible:
        return {segments[i]["id"]: segments[i]["texte"] for i in indices}
    deja = [s.get("traduction") or "" for s in segments]
    resultat: dict[int, str] = {}
    modele = get_runtime().ollama_model
    if modele != "nllb" and ollama_disponible(modele):
        try:
            with httpx.Client(base_url=get_settings().ollama_url, timeout=180) as client:
                for n, i in enumerate(indices):
                    resultat[segments[i]["id"]] = _traduire_segment(client, modele, segments, i, source, cible, deja)
                    if on_progress:
                        on_progress((n + 1) / len(indices))
            return resultat
        except (httpx.HTTPError, TranslationError) as exc:
            logger.warning("Ollama a échoué (%s) : passage à NLLB", exc)
        finally:
            _decharger_ollama(modele)
    selection = [segments[i] for i in indices]
    return dict(zip((s["id"] for s in selection), _traduire_nllb(selection, source, cible, on_progress), strict=True))


def translate_segments(
    segments: list[dict[str, Any]],
    source: str,
    cible: str,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[str], str]:
    """Traduit tous les segments. Renvoie les traductions et le moteur utilisé."""
    if not segments:
        return [], "aucun"
    if source == cible or (source.startswith("zh") and cible.startswith("zh")):
        return [s["texte"] for s in segments], "identique"

    modele = get_runtime().ollama_model
    if modele != "nllb" and ollama_disponible(modele):
        logger.info("Traduction %s → %s avec Ollama %s", source, cible, modele)
        traductions: list[str] = []
        try:
            relecture = get_settings().relecture_traduction and len(segments) > 1
            part = 0.65 if relecture else 1.0  # la relecture prend le reste de la progression
            with httpx.Client(base_url=get_settings().ollama_url, timeout=180) as client:
                for i in range(len(segments)):
                    traductions.append(
                        _traduire_segment(client, modele, segments, i, source, cible, traductions)
                    )
                    if on_progress:
                        on_progress(part * (i + 1) / len(segments))
                if relecture:
                    # Relecture du dialogue entier : cohérence (noms, tutoiement), accords, naturel
                    corrigees = 0
                    for i in range(len(segments)):
                        nouvelle = relire(client, modele, segments, traductions, i, source, cible)
                        corrigees += nouvelle != traductions[i]
                        traductions[i] = nouvelle
                        if on_progress:
                            on_progress(part + (1 - part) * (i + 1) / len(segments))
                    logger.info("Relecture : %d ligne(s) corrigée(s) sur %d", corrigees, len(segments))
            return traductions, f"ollama:{modele}"
        except (httpx.HTTPError, TranslationError) as exc:
            logger.warning("Ollama a échoué (%s) : passage à NLLB", exc)
        finally:
            _decharger_ollama(modele)

    logger.info("Traduction %s → %s avec NLLB-200 (secours)", source, cible)
    return _traduire_nllb(segments, source, cible, on_progress), "nllb"
