import pytest
from pydantic import ValidationError

from csv_sandbox_agent.schemas import ProfilerReport


def test_valid_profiler_report_passes():
    report = ProfilerReport(
        dataset_summary="Small CSV with missing values.",
        likely_task_type="unknown",
        target_column=None,
        column_issues=[
            {
                "column": "Amount",
                "issue_type": "currency_or_numeric_text",
                "severity": "medium",
                "evidence": "Sample values include $ symbols.",
                "recommended_action": "Strip currency symbols and parse numeric values.",
            }
        ],
        global_recommendations=["Avoid global dropna."],
        warnings=[],
    )
    assert report.column_issues[0].column == "Amount"


def test_invented_malformed_report_fails():
    with pytest.raises(ValidationError):
        ProfilerReport(
            dataset_summary="Bad",
            likely_task_type="clustering",
            target_column=None,
            column_issues=[
                {
                    "column": "x",
                    "issue_type": "imaginary_issue",
                    "severity": "urgent",
                    "evidence": "none",
                    "recommended_action": "none",
                }
            ],
            global_recommendations=[],
            warnings=[],
            extra_field=True,
        )
