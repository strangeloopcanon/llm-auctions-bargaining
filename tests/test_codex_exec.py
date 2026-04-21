from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.codex_exec import (
    build_codex_exec_command,
    build_output_schema,
    parse_codex_usage_from_jsonl,
    parse_last_message_json,
    run_codex_json,
)
from experiments.settings import ModelSettings


def test_build_codex_exec_command_uses_read_only_sandbox(tmp_path: Path) -> None:
    command = build_codex_exec_command(
        cwd=tmp_path,
        output_path=tmp_path / "last.txt",
        model="gpt-5.4",
        schema_path=tmp_path / "schema.json",
        reasoning_effort="medium",
    )

    assert command[0:2] == ["codex", "exec"]
    assert "-m" in command
    assert command[command.index("-m") + 1] == "gpt-5.4"
    assert "-c" in command
    assert command[command.index("-c") + 1] == 'model_reasoning_effort="medium"'
    assert "--sandbox" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command
    assert "--ephemeral" in command


def test_parse_codex_usage_from_jsonl_collects_turn_usage() -> None:
    usage = parse_codex_usage_from_jsonl(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":2,"output_tokens":3}}',
                '{"type":"turn.completed","usage":{"input_tokens":5,"cached_input_tokens":1,"output_tokens":7}}',
            ]
        )
    )

    assert usage == {"calls": 2, "input_tokens": 15, "output_tokens": 10}


def test_parse_last_message_json_requires_object() -> None:
    data = parse_last_message_json('{"launch_products":["P1"]}')
    assert data == {"launch_products": ["P1"]}


def test_build_output_schema_handles_nested_arrays_and_objects() -> None:
    schema = build_output_schema(
        {
            "type": "OBJECT",
            "properties": {
                "bids": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "feature_id": {"type": "STRING"},
                            "points": {"type": "INTEGER"},
                        },
                        "required": ["feature_id", "points"],
                    },
                },
                "commentary": {"type": "STRING"},
            },
            "required": ["bids"],
        }
    )

    assert schema == {
        "type": "object",
        "properties": {
            "bids": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature_id": {"type": "string"},
                        "points": {"type": "integer"},
                    },
                    "required": ["feature_id", "points"],
                    "additionalProperties": False,
                },
            },
            "commentary": {"type": "string"},
        },
        "required": ["bids"],
        "additionalProperties": False,
    }


def test_run_codex_json_reads_final_message_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, input, text, capture_output, check, env, timeout):  # type: ignore[override]
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text('{"commentary":"done"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4}}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    data, text, usage = run_codex_json(
        prompt="Reply with JSON.",
        model_settings=ModelSettings(provider="codex", model="gpt-5.4", codex_timeout_seconds=30),
        response_schema={"type": "OBJECT", "properties": {"commentary": {"type": "STRING"}}},
    )

    assert data == {"commentary": "done"}
    assert text == '{"commentary":"done"}'
    assert usage == {"calls": 1, "input_tokens": 12, "output_tokens": 4}


def test_run_codex_json_surfaces_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[override]
        raise subprocess.TimeoutExpired(cmd="codex", timeout=9)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 9 seconds"):
        run_codex_json(
            prompt="Reply with JSON.",
            model_settings=ModelSettings(provider="codex", model="gpt-5.4", codex_timeout_seconds=9),
        )


def test_run_codex_json_surfaces_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, input, text, capture_output, check, env, timeout):  # type: ignore[override]
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad run")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 1: bad run"):
        run_codex_json(
            prompt="Reply with JSON.",
            model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        )
