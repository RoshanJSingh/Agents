import pytest

from csv_sandbox_agent.code_validation import CodeValidationError, validate_generated_code


SAFE_SCRIPT = r'''
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df.to_csv(output_dir / "cleaned_data.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps({"model_trained": False}), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"model_trained": False}), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


def test_safe_pandas_sklearn_script_passes():
    validate_generated_code(SAFE_SCRIPT)


def test_import_os_fails():
    with pytest.raises(CodeValidationError):
        validate_generated_code(SAFE_SCRIPT.replace("import json", "import json\nimport os"))


def test_subprocess_call_fails():
    code = SAFE_SCRIPT.replace(
        "import json",
        "import json\nimport subprocess",
    ).replace(
        "df = pd.read_csv(args.input)",
        "subprocess.run(['echo', 'x'])\n    df = pd.read_csv(args.input)",
    )
    with pytest.raises(CodeValidationError):
        validate_generated_code(code)


def test_eval_fails():
    with pytest.raises(CodeValidationError):
        validate_generated_code(SAFE_SCRIPT.replace("df = pd.read_csv(args.input)", "eval('1+1')\n    df = pd.read_csv(args.input)"))


def test_missing_argparse_contract_fails():
    code = SAFE_SCRIPT.replace("parser.add_argument(\"--input\", required=True)", "")
    with pytest.raises(CodeValidationError):
        validate_generated_code(code)


def test_absolute_dangerous_path_fails():
    code = SAFE_SCRIPT.replace(
        "output_dir = Path(args.output_dir)",
        "Path('/')\n    output_dir = Path(args.output_dir)",
    )
    with pytest.raises(CodeValidationError):
        validate_generated_code(code)


def test_open_absolute_path_outside_workspace_fails():
    code = SAFE_SCRIPT.replace(
        "df = pd.read_csv(args.input)",
        "open('/etc/passwd').read()\n    df = pd.read_csv(args.input)",
    )
    with pytest.raises(CodeValidationError):
        validate_generated_code(code)


def test_workspace_parent_escape_path_fails():
    code = SAFE_SCRIPT.replace(
        "df = pd.read_csv(args.input)",
        "open('/workspace/../etc/passwd').read()\n    df = pd.read_csv(args.input)",
    )
    with pytest.raises(CodeValidationError):
        validate_generated_code(code)
