from __future__ import annotations

import json
from typing import Any

from .schemas import ProfilerReport, model_to_dict


FORBIDDEN_IMPORTS = (
    "os, subprocess, socket, requests, urllib, http, ftplib, shutil, "
    "multiprocessing, threading, asyncio"
)


SCRIPT_CONTRACT = f"""
Return a complete Python script for generated_script.py.

Hard requirements:
- Return code only. No markdown, no prose, no diff.
- The script must be self-contained.
- Allowed imports only: pandas, numpy, sklearn, joblib, json, argparse, pathlib, warnings, re, math, typing, datetime, traceback.
- Forbidden imports: {FORBIDDEN_IMPORTS}.
- Do not call shell commands, network APIs, eval, exec, compile, input, globals, locals, vars, or __import__.
- Never read or write outside /workspace and /workspace/output.
- Accept exactly these CLI arguments: --input, --output-dir, --target.
- Create the output directory.
- Always save cleaned_data.csv, metrics.json, and manifest.json.
- Save model.pkl only when a model is actually trained.
- If no target is provided, write metrics.json with model_trained false and no_model_reason.
- If the target column is missing, entirely null, or too few rows remain, skip training and explain in metrics.json.
- Do not mutate the target column as a feature.
- Use SimpleImputer, ColumnTransformer, Pipeline, and OneHotEncoder(handle_unknown="ignore") for model pipelines.
- Use train_test_split only when enough rows exist.
- Infer classification vs regression conservatively from target dtype and cardinality.
- For classification, report accuracy and f1_weighted when possible.
- For regression, report rmse, mae, and r2 when possible.
- Write manifest.json with input_path, output_files, row_count_before, row_count_after, columns_before, columns_after, model_trained, and warnings.

Data-cleaning guidance:
- Trim whitespace in text values.
- Convert null-like tokens ("", "na", "n/a", "null", "none", "nan", "-", "--", "?") to missing values.
- Do not call df.dropna globally.
- Do not invent columns.
- Do not assume target exists.
- Do not assume dates parse cleanly.
- Do not assume numeric-looking strings are valid numbers.
- For likely currency/numeric text, remove currency symbols and commas before pd.to_numeric(errors="coerce").
- For date-like columns, use pd.to_datetime(errors="coerce") and derive numeric year/month/day/dayofweek features.
- Do not one-hot encode extremely high-cardinality columns blindly; drop or safely frequency-encode them for modeling.
- Do not crash if any individual transformation fails.
""".strip()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def build_profiler_prompt(raw_profile: dict[str, Any]) -> tuple[str, str]:
    system = """
You are a conservative data-quality profiler agent.
Return JSON only. No markdown. No prose.
Your JSON must conform exactly to this schema:
{
  "dataset_summary": "string",
  "likely_task_type": "classification|regression|unknown",
  "target_column": "string or null",
  "column_issues": [
    {
      "column": "string",
      "issue_type": "missing_values|bad_date_format|currency_or_numeric_text|categorical_encoding_needed|high_cardinality|mixed_type|duplicate_column|target_problem|constant_or_near_constant|other",
      "severity": "low|medium|high",
      "evidence": "string",
      "recommended_action": "string"
    }
  ],
  "global_recommendations": ["string"],
  "warnings": ["string"]
}
Do not invent columns.
Only mention columns present in raw_profile.json.
Be conservative.
Do not recommend transformations unsupported by evidence.
If a target was not supplied in the profile, leave target_column null unless strongly evident.
""".strip()
    user = f"Analyze this deterministic Docker-generated raw CSV profile:\n\n{_json(raw_profile)}"
    return system, user


def build_engineer_prompt(
    raw_profile: dict[str, Any],
    profiler_report: ProfilerReport,
    target: str | None,
) -> tuple[str, str]:
    system = (
        "You are a senior Python data engineer. Return a full Python script only. "
        "No markdown, no prose."
    )
    user = f"""
Target column requested by user: {target or ""}

{SCRIPT_CONTRACT}

Common pitfalls to avoid:
- Do not invent columns.
- Do not assume the requested target exists.
- Do not assume dates parse cleanly.
- Do not assume all numeric-looking strings are valid numbers.
- Do not call df.dropna globally.
- Do not one-hot encode extremely high-cardinality columns blindly.
- Do not train if too few rows remain.
- Do not crash if a transformation fails.
- Do not write outside /workspace/output.
- Do not use forbidden imports.
- Do not include markdown.

Raw profile:
{_json(raw_profile)}

Validated profiler report:
{_json(model_to_dict(profiler_report))}
""".strip()
    return system, user


def build_debugger_prompt(
    code: str,
    compact_error: str,
    profiler_report: ProfilerReport,
    history: list[dict[str, Any]],
    target: str | None,
) -> tuple[str, str]:
    system = """
You are a debugger agent for a Docker-isolated generated Python script.
Return corrected full Python script only.
Do not return markdown.
Do not return a diff.
Preserve the CLI contract: --input, --output-dir, --target.
Fix the actual error.
Do not remove required outputs.
Do not introduce forbidden imports.
Do not make broad unrelated rewrites unless necessary.
If a column is missing, write defensive code rather than assuming it exists.
Use exact column names from the CSV/profile.
Avoid hardcoding sample-specific rows.
Do not repeat the same failed solution.
""".strip()
    user = f"""
Target column requested by user: {target or ""}

{SCRIPT_CONTRACT}

Compacted failure:
{compact_error}

Previous failure history:
{_json(history[-5:])}

Validated profiler report:
{_json(model_to_dict(profiler_report))}

Previous generated code:
```python
{code}
```
""".strip()
    return system, user
