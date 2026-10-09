"""Pipeline orchestration, bounded validation retries, and report helpers."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable

from json_validator import ReviewAnalysis, format_validated_json, validate_json_output
from review_analyzer import ReviewAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    analysis: ReviewAnalysis
    raw_json: str
    attempts: int


class PipelineError(RuntimeError):
    """Raised after all bounded parse/validation attempts fail."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class CsvInputError(ValueError):
    """Raised when an uploaded CSV does not match the supported input format."""


@dataclass(frozen=True)
class ReviewInput:
    review_id: str
    review: str


OutputGenerator = Callable[[str, str, int], str | dict[str, Any]]


class ReviewPipeline:
    """Connect analyzer -> JSON formatter/validator with bounded retries."""

    def __init__(
        self,
        analyzer: ReviewAnalyzer | None = None,
        max_retries: int = 2,
        output_generator: OutputGenerator | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater.")
        self.analyzer = analyzer or ReviewAnalyzer()
        self.max_retries = max_retries
        self.output_generator = output_generator

    def analyze(self, review: Any, review_id: str = "R001") -> PipelineResult:
        text = review if isinstance(review, str) else ("" if review is None else str(review))
        total_attempts = self.max_retries + 1
        last_error = "No output was produced."

        for attempt in range(1, total_attempts + 1):
            try:
                if self.output_generator is None:
                    structured = self.analyzer.analyze(text, review_id)
                    raw = format_validated_json(structured)
                else:
                    generated = self.output_generator(text, review_id, attempt)
                    raw = generated if isinstance(generated, str) else json.dumps(
                        generated, ensure_ascii=False, allow_nan=False
                    )
                parsed = validate_json_output(raw)
                # Normalize the saved output from the validated model. The raw
                # generated text is retained for transparency and test coverage.
                logger.info("Review %s passed validation on attempt %d", review_id, attempt)
                return PipelineResult(analysis=parsed, raw_json=raw, attempts=attempt)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Review %s output failed validation on attempt %d/%d (%s)",
                    review_id,
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                )

        logger.error("Review %s exhausted its %d validation attempts", review_id, total_attempts)
        raise PipelineError(
            f"Could not produce schema-valid JSON after {total_attempts} attempt(s). "
            f"Last error: {last_error}",
            attempts=total_attempts,
        )


def parse_reviews_csv(csv_text: str) -> list[ReviewInput]:
    """Parse a CSV containing a review/review_text/text column and optional IDs."""
    try:
        reader = csv.DictReader(StringIO(csv_text), strict=True)
        if not reader.fieldnames:
            raise CsvInputError("The CSV is empty or does not contain a header row.")
        headers = {header.strip().lower(): header for header in reader.fieldnames if header}
        review_column = next(
            (headers[name] for name in ("review", "review_text", "text") if name in headers),
            None,
        )
        if review_column is None:
            raise CsvInputError("Add a column named review, review_text, or text.")
        id_column = next(
            (headers[name] for name in ("review_id", "id") if name in headers),
            None,
        )
        records: list[ReviewInput] = []
        for row_index, row in enumerate(reader, start=1):
            if None in row:
                raise CsvInputError(f"CSV row {row_index + 1} has more values than the header.")
            review_id = (row.get(id_column) or "").strip() if id_column else ""
            review = row.get(review_column) or ""
            records.append(ReviewInput(
                review_id=review_id or f"R{row_index:03d}",
                review=review,
            ))
        if not records:
            raise CsvInputError("The CSV has a header but no review rows.")
        return records
    except CsvInputError:
        raise
    except (csv.Error, UnicodeError) as exc:
        raise CsvInputError(f"Could not read CSV: {exc}") from exc


def json_report(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False)


def csv_report(records: list[dict[str, Any]]) -> str:
    fields = (
        "review_id", "review", "sentiment", "confidence", "features",
        "feature_requests", "summary", "attempts", "error",
    )
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        analysis = record.get("analysis", {})
        if isinstance(analysis, ReviewAnalysis):
            analysis = analysis.model_dump(mode="json")
        writer.writerow({
            "review_id": record.get("review_id") or analysis.get("review_id", ""),
            "review": record.get("review", ""),
            "sentiment": analysis.get("sentiment", ""),
            "confidence": analysis.get("confidence", ""),
            "features": json.dumps(analysis.get("features", []), ensure_ascii=False),
            "feature_requests": json.dumps(analysis.get("feature_requests", []), ensure_ascii=False),
            "summary": analysis.get("summary", ""),
            "attempts": record.get("attempts", ""),
            "error": record.get("error", ""),
        })
    return output.getvalue()
