"""Streamlit dashboard for the End-to-End Prompt Engineering Build."""

from __future__ import annotations

import json
import logging
from typing import Any

import streamlit as st

from evaluation import evaluation_csv_report, run_evaluation
from pipeline import (
    CsvInputError,
    PipelineError,
    ReviewPipeline,
    csv_report,
    json_report,
    parse_reviews_csv,
)
from review_analyzer import suggested_improvements

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
pipeline = ReviewPipeline(max_retries=2)

st.set_page_config(page_title="Review Insight Lab", page_icon="📊", layout="wide")
st.title("End-to-End Prompt Engineering Build")
st.caption(
    "A local, deterministic review-analysis pipeline with strict JSON validation "
    "and measurable evaluation. No paid API key is required."
)


def _analysis_dict(record: dict[str, Any]) -> dict[str, Any]:
    analysis = record.get("analysis", {})
    if hasattr(analysis, "model_dump"):
        return analysis.model_dump(mode="json")
    return analysis if isinstance(analysis, dict) else {}


def _render_analysis(record: dict[str, Any], expanded: bool = True) -> None:
    analysis = _analysis_dict(record)
    if record.get("error"):
        st.error(record["error"])
        return
    if not analysis:
        return

    left, right = st.columns([1, 2])
    left.metric("Overall sentiment", analysis["sentiment"].title())
    right.metric("Heuristic confidence", f'{analysis["confidence"]:.0%}')
    st.write(analysis["summary"])

    features = analysis["features"]
    st.subheader("Product feature results")
    if features:
        st.dataframe(features, hide_index=True, width="stretch")
    else:
        st.info("No recognized product features were found in this review.")

    requests = analysis["feature_requests"]
    st.subheader("Explicit feature requests")
    if requests:
        st.dataframe(requests, hide_index=True, width="stretch")
    else:
        st.info("No explicit feature request was detected.")

    improvements = suggested_improvements(analysis)
    st.subheader("Suggested improvements")
    if improvements:
        st.dataframe(improvements, hide_index=True, width="stretch")
    else:
        st.caption("No negative feature evidence suggests a specific improvement.")

    with st.expander("Validated JSON output", expanded=expanded):
        st.code(json.dumps(analysis, ensure_ascii=False, indent=2), language="json")
    st.caption(f'Validation attempts: {record.get("attempts", 1)}')


if "analysis_records" not in st.session_state:
    st.session_state.analysis_records = []
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None

st.header("Analyze reviews")
single_review = st.text_area(
    "Product review",
    placeholder="Paste one customer review here…",
    height=150,
    key="single_review",
)
if st.button("Analyze Review", type="primary"):
    try:
        result = pipeline.analyze(single_review, "R001")
        st.session_state.analysis_records = [{
            "review_id": "R001",
            "review": single_review,
            "analysis": result.analysis.model_dump(mode="json"),
            "attempts": result.attempts,
        }]
    except PipelineError as exc:
        st.session_state.analysis_records = [{
            "review_id": "R001",
            "review": single_review,
            "error": str(exc),
            "attempts": exc.attempts,
        }]

if st.session_state.analysis_records:
    st.divider()
    st.subheader("Latest results")
    for record in st.session_state.analysis_records:
        title = record.get("review_id", "Review")
        with st.expander(title, expanded=len(st.session_state.analysis_records) == 1):
            _render_analysis(record, expanded=False)

st.header("Analyze a CSV batch")
uploaded_file = st.file_uploader("Upload a CSV with a review column", type=["csv"])
if uploaded_file is not None:
    st.caption("Accepted review columns: review, review_text, or text. An optional review_id column is supported.")
    if st.button("Analyze uploaded CSV"):
        try:
            csv_text = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
            inputs = parse_reviews_csv(csv_text)
            batch: list[dict[str, Any]] = []
            for review in inputs:
                try:
                    result = pipeline.analyze(review.review, review.review_id)
                    batch.append({
                        "review_id": review.review_id,
                        "review": review.review,
                        "analysis": result.analysis.model_dump(mode="json"),
                        "attempts": result.attempts,
                    })
                except PipelineError as exc:
                    batch.append({
                        "review_id": review.review_id,
                        "review": review.review,
                        "error": str(exc),
                        "attempts": exc.attempts,
                    })
            st.session_state.analysis_records = batch
        except CsvInputError as exc:
            st.error(str(exc))

if st.session_state.analysis_records:
    records = st.session_state.analysis_records
    st.subheader("Download analysis reports")
    download_json, download_csv = st.columns(2)
    download_json.download_button(
        "Download JSON report",
        data=json_report(records),
        file_name="review-analysis.json",
        mime="application/json",
        key="analysis-json-download",
    )
    download_csv.download_button(
        "Download CSV report",
        data=csv_report(records),
        file_name="review-analysis.csv",
        mime="text/csv",
        key="analysis-csv-download",
    )

st.divider()
st.header("Evaluate the pipeline")
st.write(
    "Run the built-in labeled cases for sentiment, feature and request extraction, "
    "input edge cases, prompt-injection resistance, and output validity."
)
if st.button("Run Evaluation"):
    st.session_state.evaluation_report = run_evaluation()

report = st.session_state.evaluation_report
if report:
    metrics = report.metrics
    st.subheader("Measured results")
    metric_columns = st.columns(4)
    metric_columns[0].metric("JSON validity", f'{metrics["json_validity_rate"]:.1%}')
    metric_columns[1].metric("Field completeness", f'{metrics["field_completeness_rate"]:.1%}')
    metric_columns[2].metric("Sentiment accuracy", f'{metrics["sentiment_accuracy"]:.1%}')
    metric_columns[3].metric(
        "Passed / failed",
        f'{metrics["passed_cases"]} / {metrics["failed_cases"]}',
        help=f'{metrics["total_cases"]} labeled cases',
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Feature precision", f'{metrics["feature_precision"]:.1%}')
    metric_columns[1].metric("Feature recall", f'{metrics["feature_recall"]:.1%}')
    metric_columns[2].metric("Request precision / recall", f'{metrics["request_precision"]:.1%} / {metrics["request_recall"]:.1%}')
    metric_columns[3].metric("Refusal rate", f'{metrics["refusal_rate"]:.1%}')

    st.subheader("Measured metrics vs. targets")
    target_rows = [
        {
            "Metric": name.replace("_", " ").title(),
            "Measured": f'{check["measured"]:.1%}',
            "Target": f'{check["target"]:.1%}' if name != "refusal_rate" else "≤ 0.0%",
            "Target met": "Yes" if check["met"] else "No",
        }
        for name, check in report.targets.items()
    ]
    st.dataframe(target_rows, hide_index=True, width="stretch")

    st.subheader("Failed test cases")
    if report.failed_cases:
        failure_rows = [{
            "Case": case["name"],
            "Expected sentiment": case["expected_sentiment"],
            "Predicted sentiment": case["predicted_sentiment"] or "No result",
            "Issues": " ".join(case["issues"]),
        } for case in report.failed_cases]
        st.dataframe(failure_rows, hide_index=True, width="stretch")
    else:
        st.success("All labeled cases passed in this measured run.")

    st.subheader("Suggested fixes")
    for suggestion in report.suggested_fixes:
        st.write(f"- {suggestion}")

    evaluation_json, evaluation_csv = st.columns(2)
    evaluation_json.download_button(
        "Download evaluation JSON",
        data=json_report([report.to_dict()]),
        file_name="evaluation-report.json",
        mime="application/json",
        key="evaluation-json-download",
    )
    evaluation_csv.download_button(
        "Download evaluation CSV",
        data=evaluation_csv_report(report),
        file_name="evaluation-cases.csv",
        mime="text/csv",
        key="evaluation-csv-download",
    )
