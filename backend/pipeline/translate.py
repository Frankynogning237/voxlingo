"""Traduction contextuelle locale via Ollama (Qwen 2.5), repli sur NLLB-200.

Chaque segment est traduit avec le contexte de ses voisins (avant et après),
en visant une longueur proche de l'original pour faciliter le doublage.
"""

from __future__ import annotations

import gc
import json
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
    """Vrai si le serveur Ollama répond et que le modèle est téléchargé."""
    url = get_settings().ollama_url
    try:
        tags = httpx.get(f"{url}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        logger.warning("Ollama ne répond pas sur %s", url)
        return False
    noms = {m.get("name") for m in tags.get("models", [])}
    if modele not in noms and f"{modele}:latest" not in noms:
        logger.warning("Modèle %s absent d'Ollama (ollama pull %s)", modele, modele)
        return False
    return True


SYSTEM_PROMPT = (
    "You are a professional subtitle and dubbing translator. You translate ONE line of a video "
    "transcript at a time. Rules: (1) keep the exact meaning; (2) keep the register and tone "
    "(casual, neutral, formal); (3) replace idioms with a natural equivalent, never word for word; "
    "(4) keep the length close to the original so the dubbed voice fits the same time; "
    "(5) translate ONLY the line to translate, never the context lines; "
    "(6) the output must be written entirely in the target language. "
    'Answer with strict JSON only: {"translation": "..."}'
)


def _prompt(segments: list[dict[str, Any]], i: int, source: str, cible: str,
            deja_traduits: list[str], avec_contexte: bool = True) -> str:
    seg = segments[i]
    syllabes = syllabes_cibles(seg["texte"], source, cible)
    lignes = [
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
    ]
    if avec_contexte:
        avant = [
            f'- "{segments[j]["texte"]}" -> "{deja_traduits[j]}"'
            for j in range(max(0, i - CONTEXTE), i)
        ]
        apres = [f'- "{segments[j]["texte"]}"' for j in range(i + 1, min(len(segments), i + 1 + CONTEXTE))]
        if avant:
            lignes += ["", "Previous lines (already translated, for context only):", *avant]
        if apres:
            lignes += ["", "Next lines (for context only, do not translate):", *apres]
    lignes += [
        "",
        f'LINE TO TRANSLATE: "{seg["texte"]}"',
        f"Target length: about {syllabes} syllables "
        f"(between {max(1, round(syllabes * 0.85))} and {round(syllabes * 1.15)}).",
    ]
    return "\n".join(lignes)


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


SHORTEN_PROMPT = (
    "You shorten dubbing translations so the dubbed voice fits the original timing. "
    "Rewrite the translation more concisely in the same target language: keep the meaning, "
    "the tone and a natural spoken style, drop filler words, prefer shorter synonyms. "
    'Answer with strict JSON only: {"translation": "..."}'
)


def raccourcir(client: httpx.Client, modele: str, original: str, traduction: str,
               source: str, cible: str, syllabes_max: int) -> str:
    """Demande au modèle une version plus courte de la traduction (au plus `syllabes_max` syllabes)."""
    prompt = "\n".join([
        f"Source language: {LANGUAGE_NAMES.get(source, source)}",
        f"Target language: {LANGUAGE_NAMES.get(cible, cible)}",
        f'Original line: "{original}"',
        f'Current translation (too long): "{traduction}"',
        f"Maximum length: {syllabes_max} syllables.",
    ])
    reponse = client.post(
        "/api/chat",
        json={
            "model": modele,
            "messages": [
                {"role": "system", "content": SHORTEN_PROMPT},
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
        },
    )
    reponse.raise_for_status()
    try:
        texte = json.loads(reponse.json()["message"]["content"])["translation"].strip().strip('"').strip()
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise TranslationError("Réponse inattendue du modèle pendant le raccourcissement.") from exc
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
            with httpx.Client(base_url=get_settings().ollama_url, timeout=180) as client:
                for i in range(len(segments)):
                    traductions.append(
                        _traduire_segment(client, modele, segments, i, source, cible, traductions)
                    )
                    if on_progress:
                        on_progress((i + 1) / len(segments))
            return traductions, f"ollama:{modele}"
        except (httpx.HTTPError, TranslationError) as exc:
            logger.warning("Ollama a échoué (%s) : passage à NLLB", exc)
        finally:
            _decharger_ollama(modele)

    logger.info("Traduction %s → %s avec NLLB-200 (secours)", source, cible)
    return _traduire_nllb(segments, source, cible, on_progress), "nllb"
