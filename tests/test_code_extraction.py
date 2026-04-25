import pytest

from csv_sandbox_agent.code_extraction import CodeExtractionError, extract_python_code


def test_raw_code_parses():
    code = "import argparse\n\ndef main():\n    return 1\n\nmain()\n"
    assert extract_python_code(code).startswith("import argparse")


def test_fenced_python_block_parses():
    response = "```python\nimport json\n\ndef main():\n    print(json.dumps({}))\n\nmain()\n```"
    assert "import json" in extract_python_code(response)


def test_prose_before_fenced_block_parses():
    response = "Here is your code:\n\n```python\nimport argparse\n\ndef main():\n    pass\n\nmain()\n```"
    assert extract_python_code(response).startswith("import argparse")


def test_multiple_code_blocks_choose_largest_python_block():
    response = """
```python
print("small")
```

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    return parser

main()
```
"""
    extracted = extract_python_code(response)
    assert "ArgumentParser" in extracted
    assert "small" not in extracted


def test_invalid_code_raises():
    with pytest.raises(CodeExtractionError):
        extract_python_code("```python\nif broken\n```")
