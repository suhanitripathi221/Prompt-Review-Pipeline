"""Strict JSON parsing and Pydantic validation for analyzer output."""

from __future__ import annotations

import json
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["positive", "negative", "neutral"]
Priority = Literal["high", "medium", "low"]


class FeatureMention(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    feature: str = Field(min_length=1)
    sentiment: Sentiment
    evidence: str = Field(min_length=1)


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    feature: str = Field(min_length=1)
    request: str = Field(min_length=1)
    priority: Priority


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    review_id: str = Field(min_length=1)
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    features: list[FeatureMention]
    feature_requests: list[FeatureRequest]
    summary: str = Field(min_length=1)


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_json_output(raw: str | dict[str, Any]) -> ReviewAnalysis:
    """Parse JSON (or a structured mapping) and enforce the full schema.

    Standard JSON parsing rejects single quotes and trailing commas. Duplicate
    keys and non-standard NaN/Infinity values are also rejected explicitly.
    """
    if isinstance(raw, str):
        payload = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise TypeError("Analyzer output must be a JSON string or object.")
    return ReviewAnalysis.model_validate(payload, strict=True)


def format_validated_json(value: ReviewAnalysis | dict[str, Any]) -> str:
    """Return indented, standards-compliant JSON after schema validation."""
    validated = value if isinstance(value, ReviewAnalysis) else ReviewAnalysis.model_validate(value, strict=True)
    return json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False)
