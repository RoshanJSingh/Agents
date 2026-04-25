from __future__ import annotations


PROFILER_SCRIPT = r'''
import csv
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("/workspace/input.csv")
OUTPUT_PATH = Path("/workspace/raw_profile.json")
NULL_LIKE = {"", "na", "n/a", "null", "none", "nan", "-", "--", "?"}


def safe_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return str(value)


def decode_sample(path):
    raw = path.read_bytes()[:65536]
    errors = []
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return raw.decode(enc), enc, errors
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc}")
    return raw.decode("latin-1", errors="replace"), "latin-1", errors


def detect_delimiter(sample):
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter, []
    except Exception as exc:
        return ",", [f"delimiter detection failed, defaulted to comma: {exc}"]


def raw_header(path, encoding, delimiter):
    try:
        with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            return next(reader, [])
    except Exception:
        return []


def line_count(path):
    try:
        with path.open("rb") as handle:
            total = sum(1 for _ in handle)
        return max(total - 1, 0)
    except Exception:
        return None


def read_sample(path, encoding, delimiter):
    read_warnings = []
    errors = []
    attempts = [
        {"sep": delimiter, "engine": "python"},
        {"sep": None, "engine": "python"},
        {"sep": ",", "engine": "python"},
    ]
    for kwargs in attempts:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    nrows=50,
                    on_bad_lines="warn",
                    **kwargs,
                )
            read_warnings.extend(str(item.message) for item in caught)
            return df, read_warnings, errors, kwargs
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), read_warnings, ["empty CSV"], kwargs
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(), read_warnings, errors, attempts[-1]


def looks_currency_like(series):
    values = series.dropna().astype(str).head(50)
    if values.empty:
        return False, 0.0
    hits = values.str.contains(r"[$₹€£,%]", regex=True).sum()
    ratio = float(hits) / float(len(values))
    return ratio >= 0.2, ratio


def looks_date_like(series):
    values = series.dropna().astype(str).head(50)
    if len(values) < 3:
        return False, 0.0
    parsed = pd.to_datetime(values, errors="coerce", infer_datetime_format=True)
    ratio = float(parsed.notna().sum()) / float(len(values))
    return ratio >= 0.6, ratio


def mixed_type_summary(series):
    values = series.dropna().head(50)
    if values.empty:
        return {"mixed": False, "kinds": []}
    kinds = set()
    for value in values.astype(str):
        stripped = value.strip()
        if stripped.lower() in NULL_LIKE:
            kinds.add("null_like")
        elif re.fullmatch(r"[-+]?\d+(\.\d+)?", stripped.replace(",", "")):
            kinds.add("numeric_text")
        elif pd.notna(pd.to_datetime(stripped, errors="coerce")):
            kinds.add("date_text")
        else:
            kinds.add("text")
    non_null_kinds = sorted(k for k in kinds if k != "null_like")
    return {"mixed": len(non_null_kinds) > 1, "kinds": non_null_kinds}


def column_profile(df, column):
    series = df[column]
    missing = int(series.isna().sum())
    non_null = series.dropna()
    samples = [safe_json_value(v) for v in non_null.head(5).tolist()]
    numeric_stats = None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() > 0:
        numeric_stats = {
            "count": int(numeric.notna().sum()),
            "mean": safe_json_value(numeric.mean()),
            "std": safe_json_value(numeric.std()),
            "min": safe_json_value(numeric.min()),
            "max": safe_json_value(numeric.max()),
        }
    date_like, date_ratio = looks_date_like(series)
    currency_like, currency_ratio = looks_currency_like(series)
    unique = int(series.nunique(dropna=True))
    high_cardinality = bool(unique > 20 and len(series) > 0 and unique / max(len(series), 1) > 0.8)
    return {
        "dtype": str(series.dtype),
        "missing_count": missing,
        "missing_percentage": float(missing / max(len(series), 1)),
        "unique_count": unique,
        "sample_values": samples,
        "numeric_stats": numeric_stats,
        "date_like": date_like,
        "date_parse_ratio": date_ratio,
        "currency_like": currency_like,
        "currency_symbol_ratio": currency_ratio,
        "mixed_type": mixed_type_summary(series),
        "high_cardinality_sample": high_cardinality,
    }


def main():
    profile = {
        "input_path": str(INPUT_PATH),
        "profiler_version": 1,
        "encoding": None,
        "delimiter": None,
        "row_count_estimate": None,
        "sample_row_count": 0,
        "columns": [],
        "raw_header": [],
        "duplicate_header_names": [],
        "pandas_dtypes": {},
        "columns_profile": {},
        "date_like_columns": [],
        "currency_like_columns": [],
        "mixed_type_columns": [],
        "high_cardinality_columns": [],
        "warnings": [],
        "read_errors": [],
    }
    try:
        sample, encoding, decode_errors = decode_sample(INPUT_PATH)
        delimiter, delimiter_warnings = detect_delimiter(sample)
        profile["encoding"] = encoding
        profile["delimiter"] = delimiter
        profile["warnings"].extend(decode_errors)
        profile["warnings"].extend(delimiter_warnings)
        header = raw_header(INPUT_PATH, encoding, delimiter)
        profile["raw_header"] = header[:500]
        seen = {}
        duplicates = []
        for name in header:
            seen[name] = seen.get(name, 0) + 1
            if seen[name] == 2:
                duplicates.append(name)
        profile["duplicate_header_names"] = duplicates
        df, read_warnings, read_errors, read_kwargs = read_sample(INPUT_PATH, encoding, delimiter)
        profile["read_kwargs"] = read_kwargs
        profile["warnings"].extend(read_warnings)
        profile["read_errors"].extend(read_errors)
        profile["row_count_estimate"] = line_count(INPUT_PATH)
        profile["sample_row_count"] = int(len(df))
        profile["columns"] = [str(c) for c in df.columns.tolist()[:500]]
        profile["pandas_dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
        for column in df.columns[:500]:
            info = column_profile(df, column)
            profile["columns_profile"][str(column)] = info
            if info["date_like"]:
                profile["date_like_columns"].append(str(column))
            if info["currency_like"]:
                profile["currency_like_columns"].append(str(column))
            if info["mixed_type"]["mixed"]:
                profile["mixed_type_columns"].append(str(column))
            if info["high_cardinality_sample"]:
                profile["high_cardinality_columns"].append(str(column))
    except Exception as exc:
        profile["read_errors"].append(f"{type(exc).__name__}: {exc}")
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
'''
