"""Deterministic local review analysis for the capstone pipeline.

This baseline deliberately treats review text as data. It does not execute,
follow, or answer instructions that happen to appear inside a review.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "battery": ("battery", "battery life", "battery backup", "charging"),
    "camera": ("camera", "cameras", "photo quality", "photos"),
    "screen": ("screen", "display", "touchscreen"),
    "audio": ("speaker", "speakers", "sound", "audio", "headphones"),
    "performance": ("performance", "speed", "processor", "lag", "loading"),
    "price": ("price", "cost", "value", "expensive", "affordable"),
    "design": ("design", "build", "materials", "appearance"),
    "software": ("software", "operating system", "os", "interface"),
    "app": ("app", "application", "apps"),
    "customer support": ("customer support", "support", "service team"),
    "shipping": ("shipping", "delivery", "shipment"),
    "storage": ("storage", "memory", "space"),
    "keyboard": ("keyboard", "keypad", "keys"),
    "durability": ("durability", "durable", "sturdy", "fragile"),
    "connectivity": ("wifi", "wi-fi", "bluetooth", "connection", "connectivity"),
}

POSITIVE_WORDS = {
    "amazing", "awesome", "awsome", "beautiful", "best", "bright", "clean",
    "clear", "comfortable", "crisp", "excellent", "fantastic", "fast", "fine",
    "good", "great", "happy", "impressive", "love", "loved", "lovely", "perfect",
    "reliable", "sharp", "smooth", "solid", "strong", "useful", "vivid", "wonderful",
    "works", "worked", "wow", "dope", "lit", "sick",
}
NEGATIVE_WORDS = {
    "awful", "bad", "baaad", "blurry", "broken", "buggy", "crashes", "crashed",
    "disappointing", "drains", "drained", "dull", "expensive", "flimsy", "horrible",
    "laggy", "poor", "slow", "terrible", "tinny", "unusable", "worse", "worst",
    "annoying", "hate", "hated", "sucks", "suck", "trash", "overheats", "overheating",
}
NEGATORS = {"not", "never", "no", "hardly", "barely", "isn't", "wasn't", "doesn't", "can't"}
AMBIGUITY_CUES = ("i guess", "kind of", "sort of", "maybe", "not sure", "okay-ish")
REQUEST_CUE = re.compile(
    r"\b(?:please\s+(?:add|include|offer|improve|increase|make)|"
    r"(?:i|we)\s+(?:want|need|wish|would like|would love)|"
    r"would love|wish we had|should\s+(?:have|offer|include|add)|"
    r"needs?\s+(?:more|better|longer|improved|to)|"
    r"(?:add|include|offer|increase|improve)\s+(?:more|better|an?|the)?)\b",
    re.IGNORECASE,
)
HIGH_PRIORITY_CUE = re.compile(r"\b(?:urgent|critical|must|immediately|asap)\b", re.IGNORECASE)
LOW_PRIORITY_CUE = re.compile(r"\b(?:nice to have|would be nice|optional|if possible)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:['’][a-zA-Z]+)*")
SENTENCE_RE = re.compile(r"[^.!?;\n]+(?:[.!?;]+(?=\s|$)|$)")


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in WORD_RE.finditer(text)]


def _feature_is_mentioned(sentence: str, feature: str) -> bool:
    tokens = _word_tokens(sentence)
    token_set = set(tokens)
    for alias in FEATURE_ALIASES[feature]:
        alias_tokens = _word_tokens(alias)
        if not alias_tokens:
            continue
        if len(alias_tokens) == 1 and alias_tokens[0] in token_set:
            return True
        if len(alias_tokens) > 1:
            phrase = r"\b" + r"\s+".join(re.escape(token) for token in alias_tokens) + r"\b"
            if re.search(phrase, sentence, re.IGNORECASE):
                return True
        # Catch common typos (e.g. "batery", "camra") without treating short
        # words as product features.
        for word in tokens:
            for alias_token in alias_tokens:
                if len(alias_token) >= 5 and len(word) >= 4:
                    if SequenceMatcher(None, word, alias_token).ratio() >= 0.84:
                        return True
    return False


def _sentiment_score(text: str) -> int:
    tokens = _word_tokens(text)
    score = 0
    for index, token in enumerate(tokens):
        value = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if token == "fast" and any(
            word in {"drain", "drains", "drained", "die", "dies", "runs"}
            for word in tokens[max(0, index - 3):index]
        ):
            value = -1
        if value:
            prior = tokens[max(0, index - 3):index]
            if any(word in NEGATORS for word in prior):
                value *= -1
            score += value
    lowered = text.lower()
    if any(cue in lowered for cue in AMBIGUITY_CUES) and abs(score) <= 1:
        return 0
    return score


def _feature_score(sentence: str, feature: str) -> int:
    """Score the clause containing a feature so mixed feedback stays distinct."""
    clauses = re.split(r"\b(?:but|however|yet|although|though)\b", sentence, flags=re.IGNORECASE)
    matching_clauses = [clause for clause in clauses if _feature_is_mentioned(clause, feature)]
    if not matching_clauses:
        return _sentiment_score(sentence)
    strongest_clause = max(matching_clauses, key=lambda clause: abs(_sentiment_score(clause)))
    return _sentiment_score(strongest_clause)


def _sentiment(score: int) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def _priority(sentence: str) -> str:
    if HIGH_PRIORITY_CUE.search(sentence):
        return "high"
    if LOW_PRIORITY_CUE.search(sentence):
        return "low"
    return "medium"


def _summary(sentiment: str, features: list[dict[str, str]], review: str) -> str:
    if not review.strip():
        return "No review text was provided."
    if not features:
        if sentiment == "neutral":
            return "The review contains no clear sentiment about a recognized product feature."
        return f"The review is {sentiment}, but no recognized product feature was identified."

    labels = {"positive": "praises", "negative": "criticizes", "neutral": "mentions"}
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(feature["sentiment"], [])
        if feature["feature"] not in groups[feature["sentiment"]]:
            groups[feature["sentiment"]].append(feature["feature"])
    details = [
        f"{labels[tone]} {', '.join(names[:3])}"
        for tone, names in groups.items()
    ]
    return f"The review is {sentiment} overall and {', '.join(details)}."


class ReviewAnalyzer:
    """Extract sentiment, feature evidence, and explicit feature requests."""

    def analyze(self, review: Any, review_id: str = "R001") -> dict[str, Any]:
        text = review if isinstance(review, str) else ("" if review is None else str(review))
        feature_rows: list[dict[str, str]] = []
        feature_keys: set[tuple[str, str, str]] = set()
        request_rows: list[dict[str, str]] = []
        request_keys: set[tuple[str, str]] = set()
        total_score = 0
        positive_count = 0
        negative_count = 0

        for sentence in _sentences(text):
            score = _sentiment_score(sentence)
            total_score += score
            positive_count += max(score, 0)
            negative_count += max(-score, 0)
            mentioned = [
                feature for feature in FEATURE_ALIASES
                if _feature_is_mentioned(sentence, feature)
            ]
            for feature in mentioned:
                tone = _sentiment(_feature_score(sentence, feature))
                key = (feature, tone, sentence)
                if key not in feature_keys:
                    feature_rows.append({
                        "feature": feature,
                        "sentiment": tone,
                        "evidence": sentence,
                    })
                    feature_keys.add(key)

            if REQUEST_CUE.search(sentence):
                for feature in mentioned:
                    key = (feature, sentence)
                    if key not in request_keys:
                        request_rows.append({
                            "feature": feature,
                            "request": sentence,
                            "priority": _priority(sentence),
                        })
                        request_keys.add(key)

        overall_sentiment = _sentiment(total_score)
        sentiment_evidence_count = positive_count + negative_count
        if sentiment_evidence_count:
            strength = abs(total_score) / sentiment_evidence_count
            confidence = min(0.96, 0.56 + (0.34 * strength) + min(len(feature_rows), 4) * 0.025)
        else:
            confidence = 0.52 if text.strip() else 0.50
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        result = {
            "review_id": str(review_id),
            "sentiment": overall_sentiment,
            "confidence": confidence,
            "features": feature_rows,
            "feature_requests": request_rows,
            "summary": _summary(overall_sentiment, feature_rows, text),
        }
        logger.debug("Analyzed review %s: %d features, %d explicit requests", review_id, len(feature_rows), len(request_rows))
        return result


def suggested_improvements(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """Create transparent UI suggestions from negative feature evidence."""
    improvements: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in analysis.get("features", []):
        feature = item.get("feature", "")
        if item.get("sentiment") == "negative" and feature and feature not in seen:
            seen.add(feature)
            priority = "high" if any(
                marker in item.get("evidence", "").lower()
                for marker in ("terrible", "awful", "broken", "unusable", "worst", "baaad")
            ) else "medium"
            improvements.append({
                "feature": feature,
                "suggestion": f"Investigate and improve {feature} based on the negative evidence.",
                "priority": priority,
            })
    return improvements
