# csv-sandbox-agent

`csv-sandbox-agent` is a local Python 3.11+ CLI that uses a Docker sandbox and explicit LLM agents to clean messy CSV files and optionally train a baseline scikit-learn model. The orchestrator profiles the CSV inside Docker, asks an LLM for a structured data-quality report, asks another LLM for a complete cleaning/training script, validates that script, and executes it only inside a restricted container. Failures are sent to a debugger agent through a bounded retry loop.

The package name is `csv-sandbox-agent`; the import package is `csv_sandbox_agent`.

## Security Model

This project implements defense in depth:

1. Prompts constrain LLM behavior.
2. Code extraction refuses non-Python responses.
3. AST validation blocks obvious unsafe imports and calls.
4. Generated code is only executed in Docker.
5. Runtime containers have no network access and only mount the single run workspace at `/workspace`.
6. CPU, memory, process, capability, read-only-root-filesystem, and timeout limits reduce blast radius.
7. Retry circuit breakers stop repeated or low-signal failures.

Docker sandboxing reduces risk, but it is not a perfect security boundary. Keep Docker updated. Never run `generated_script.py` manually on the host. Do not mount broad host directories. Do not use this with highly sensitive data unless you understand the privacy implications of sending metadata and capped sample values to an LLM provider.

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

Configure an OpenAI-compatible provider:

```bash
export OPENAI_API_KEY=...
export CSV_SANDBOX_AGENT_MODEL=gpt-4.1-mini
# optional:
export CSV_SANDBOX_AGENT_BASE_URL=https://your-compatible-endpoint/v1
```

## Build The Sandbox Image

The CLI builds the image automatically when needed. You can also build it through:

```bash
python -m csv_sandbox_agent doctor
```

The image is tagged `csv-sandbox-agent:latest`.

## Doctor

```bash
python -m csv_sandbox_agent doctor
```

This checks Docker reachability, sandbox image availability, required Python dependencies, and LLM configuration. To skip the API key check in mock mode:

```bash
python -m csv_sandbox_agent doctor --fake-llm
```

## Profile A CSV

```bash
python -m csv_sandbox_agent profile \
  --input ./examples/messy_classification.csv \
  --workspace ./runs
```

This creates a run directory and writes `raw_profile.json`. Profiling is deterministic and runs inside Docker; the raw CSV is not sent directly to the LLM.

## Run The Full Pipeline

Cleaning only:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_classification.csv \
  --workspace ./runs \
  --max-retries 5
```

Cleaning plus baseline model training:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_regression.csv \
  --workspace ./runs \
  --target SalePrice \
  --max-retries 5
```

For a deterministic provider-free smoke run:

```bash
python -m csv_sandbox_agent run \
  --input ./examples/messy_classification.csv \
  --workspace ./runs \
  --fake-llm
```

## Run Directory

Each execution creates a directory such as:

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

The original CSV is never modified in place.

## Retries And Debugging

Every generated-code attempt is audited. The run directory stores raw LLM responses, extracted scripts, validation results, execution logs, compacted errors, debugger input summaries, and failure fingerprints.

The orchestrator stops after the retry budget or when a circuit breaker fires:

- repeated normalized failure fingerprint
- identical generated code hash after a failed attempt
- invalid Python twice in a row
- Docker timeout twice
- successful exit with missing required outputs twice
- empty compacted error with no actionable signal

## Outputs

The generated script must always write:

- `output/cleaned_data.csv`
- `output/metrics.json`
- `output/manifest.json`

It writes `output/model.pkl` only when supervised training succeeds. If no target is supplied or training is not possible, `metrics.json` contains `model_trained: false` and a clear `no_model_reason`.

## Limitations

The LLM receives profile metadata and capped sample values, not the full raw CSV. Generated scripts are validated before execution, but validation is only a risk-reduction layer. Docker configuration and host kernel security still matter. The baseline model is intentionally simple and meant for quick diagnostics, not production ML deployment.

## Tests

```bash
pytest
```

Docker-dependent tests should be marked with `@pytest.mark.docker` and skipped automatically when Docker is unavailable.

# Agents
