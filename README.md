# csv-sandbox-agent

A command line tool that cleans messy CSV files with the help of LLM agents, and can also train a simple scikit-learn model on the result. Any code the LLM writes is run only inside a locked-down Docker container, never on your machine directly.

It needs Python 3.11 or newer. The package is installed as `csv-sandbox-agent` and imported as `csv_sandbox_agent`.

## What happens in a run

1. The CSV is profiled inside Docker. The raw file is never sent to the LLM.
2. One LLM reads the profile and writes a data quality report.
3. A second LLM writes a full cleaning (and optional training) script.
4. The script is checked, then run inside a restricted container.
5. If it fails, the error goes to a debugger agent that tries to fix the script. This repeats up to a set number of times.

## Safety

No single check is trusted on its own, so there are several layers:

1. The prompts limit what the LLM is asked to do.
2. Responses that are not Python code are rejected.
3. The script's syntax tree is checked for unsafe imports and calls.
4. Generated code only ever runs in Docker.
5. The container has no network access and can only see the run folder, mounted at `/workspace`.
6. CPU, memory, process count, capabilities and run time are all limited, and the root filesystem is read-only.
7. The retry loop stops early when it keeps hitting the same failure.

Docker makes this much safer, but it is not a perfect security boundary. Keep Docker up to date, never run `generated_script.py` yourself on the host, and do not mount large folders from your machine. The LLM does see column metadata and a few capped sample values, so think twice before using it on sensitive data.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Point it at any OpenAI-compatible provider:

```bash
export OPENAI_API_KEY=...
export CSV_SANDBOX_AGENT_MODEL=gpt-4.1-mini
# optional:
export CSV_SANDBOX_AGENT_BASE_URL=https://your-compatible-endpoint/v1
```

## The sandbox image

The tool builds the Docker image (`csv-sandbox-agent:latest`) by itself the first time it is needed. The `doctor` command also builds it.

## Checking your setup

```bash
python -m csv_sandbox_agent doctor
```

This checks that Docker is reachable, the image exists, the Python dependencies are installed and the LLM settings are present. Add `--fake-llm` to skip the API key check.

## Profiling a CSV

```bash
python -m csv_sandbox_agent profile \
  --input ./examples/messy_classification.csv \
  --workspace ./runs
```

This creates a run folder and writes `raw_profile.json` to it. Profiling gives the same result every time and runs inside Docker.

## Running the full pipeline

Cleaning only:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_classification.csv \
  --workspace ./runs \
  --max-retries 5
```

Cleaning plus a baseline model:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_regression.csv \
  --workspace ./runs \
  --target SalePrice \
  --max-retries 5
```

A quick run with no LLM provider at all:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_classification.csv \
  --workspace ./runs \
  --fake-llm
```

## The run folder

Each run gets its own folder, for example:

```text
runs/run_2026_04_24_153000_ab12cd/
  input.csv
  raw_profile.json
  profiler_report.json
  generated_script.py
  generated_script_attempt_0.py
  execution_attempt_0.stdout.log
  execution_attempt_0.stderr.log
  debug_history.jsonl
  failure_fingerprints.json
  output/
    cleaned_data.csv
    model.pkl
    metrics.json
    manifest.json
```

Your original CSV is never changed.

## Retries

Every attempt is saved: the raw LLM reply, the extracted script, the validation result, the logs, a short version of the error, what the debugger was given, and a fingerprint of the failure.

The loop stops when the retry budget runs out, or earlier if:

- the same failure shows up again
- the new script is identical to the one that just failed
- the LLM returns invalid Python twice in a row
- Docker times out twice
- the script exits cleanly but the required outputs are missing, twice
- the error has nothing useful in it to act on

## Outputs

The generated script always has to write:

- `output/cleaned_data.csv`
- `output/metrics.json`
- `output/manifest.json`

`output/model.pkl` is only written when training works. If there is no target, or training is not possible, `metrics.json` says `model_trained: false` and gives the reason in `no_model_reason`.

## Limitations

The LLM only sees profile metadata and a few sample values, not the whole CSV. Scripts are checked before they run, but that check only lowers the risk. How Docker is set up, and the host kernel, still matter. The baseline model is kept simple on purpose. It is meant for a quick look at the data, not for production.

## Tests

```bash
pytest
```

Tests that need Docker are marked `@pytest.mark.docker` and are skipped when Docker is not available.
