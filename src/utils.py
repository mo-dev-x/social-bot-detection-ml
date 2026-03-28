"""Shared helpers for the bot detection pipeline."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


ENGLISH_HINTS = {"the", "and", "for", "with", "that", "this", "you", "are", "not", "have"}
FRENCH_HINTS = {"les", "des", "pour", "avec", "vous", "dans", "une", "pas", "est", "sur"}
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json_dataset(filepath: str | Path) -> dict:
    with Path(filepath).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_timestamp(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def shannon_entropy(values: Sequence) -> float:
    if not values:
        return 0.0
    if len(set(values)) <= 1:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text)


def normalize_for_similarity(text: str) -> str:
    normalized = " ".join(text.lower().split())
    normalized = URL_RE.sub(" ", normalized)
    return normalized.strip()


def estimate_language(texts: Iterable[str]) -> str:
    joined = " ".join(texts).lower()
    if not joined.strip():
        return "unknown"
    words = tokenize_words(joined)
    if not words:
        return "unknown"
    english_hits = sum(word in ENGLISH_HINTS for word in words)
    french_hits = sum(word in FRENCH_HINTS for word in words)
    if english_hits == french_hits == 0:
        accents = sum(char in "éèàùâêîôûçëïü" for char in joined)
        return "fr" if accents > 0 else "unknown"
    return "en" if english_hits >= french_hits else "fr"


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize_words(left))
    right_tokens = set(tokenize_words(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return safe_divide(len(left_tokens & right_tokens), len(left_tokens | right_tokens))
