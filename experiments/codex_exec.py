from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .settings import ModelSettings


_USAGE_KEYS = ("input_tokens", "output_tokens", "cached_input_tokens")


def build_codex_exec_command(
    *,
    cwd: Path,
    output_path: Path,
    model: str,
    schema_path: Optional[Path],
    reasoning_effort: Optional[str],
) -> list[str]:
    command = [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "-C",
        str(cwd),
        "-o",
        str(output_path),
        "-",
    ]
    if reasoning_effort:
        command[2:2] = ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
    if schema_path is not None:
        command[2:2] = ["--output-schema", str(schema_path)]
    if model:
        command[2:2] = ["-m", model]
    return command


def parse_codex_usage_from_jsonl(text: str) -> Optional[Dict[str, int]]:
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
    }
    saw_usage = False

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("type") != "turn.completed":
            continue

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue

        saw_usage = True
        totals["calls"] += 1
        for key in _USAGE_KEYS:
            value = usage.get(key, 0)
            try:
                totals[key] += int(value or 0)
            except (TypeError, ValueError):
                continue

    if not saw_usage:
        return None

    return {
        "calls": int(totals["calls"]),
        "input_tokens": int(totals["input_tokens"]),
        "output_tokens": int(totals["output_tokens"]),
    }


def parse_last_message_json(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("Codex returned an empty final message.")
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("Codex final message must be a JSON object.")
    return data


def _schema_type_to_json_schema(schema_type: Optional[str]) -> Dict[str, Any]:
    normalized = str(schema_type or "").strip().lower()
    if normalized == "array":
        return {"type": "array", "items": {"type": "string"}}
    if normalized == "object":
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    if normalized == "number":
        return {"type": "number"}
    if normalized == "integer":
        return {"type": "integer"}
    if normalized == "boolean":
        return {"type": "boolean"}
    return {"type": "string"}


def _build_json_schema(node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return {"type": "string"}

    node_type = str(node.get("type") or "").strip().lower()

    if node_type == "array":
        items = node.get("items")
        return {
            "type": "array",
            "items": _build_json_schema(items) if isinstance(items, dict) else {"type": "string"},
        }

    if node_type == "object" or isinstance(node.get("properties"), dict):
        properties = node.get("properties")
        json_properties: Dict[str, Any] = {}
        if isinstance(properties, dict):
            json_properties = {key: _build_json_schema(value) for key, value in properties.items()}

        required = node.get("required")
        if isinstance(required, list) and required:
            required_fields = [str(field) for field in required if str(field).strip()]
        else:
            required_fields = sorted(json_properties.keys())

        return {
            "type": "object",
            "properties": json_properties,
            "required": required_fields,
            "additionalProperties": False,
        }

    return _schema_type_to_json_schema(node_type)


def build_output_schema(response_schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(response_schema, dict):
        return None
    properties = response_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    return _build_json_schema(response_schema)


def _failure_message(proc: subprocess.CompletedProcess[str]) -> str:
    for raw_line in str(proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "error":
            message = str(payload.get("message") or "").strip()
            if message:
                return message

    stderr = str(proc.stderr or "").strip()
    if stderr:
        return stderr
    stdout_lines = [line.strip() for line in str(proc.stdout or "").splitlines() if line.strip()]
    if stdout_lines:
        return stdout_lines[-1]
    return "no Codex error details were returned"


def run_codex_json(
    *,
    prompt: str,
    model_settings: ModelSettings,
    response_schema: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], str, Optional[Dict[str, int]]]:
    output_schema = build_output_schema(response_schema)

    with tempfile.TemporaryDirectory(prefix="codex-ip-market-") as temp_dir:
        cwd = Path(temp_dir)
        output_path = cwd / "codex_last_message.txt"
        schema_path = None
        if output_schema is not None:
            schema_path = cwd / "codex_output_schema.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )

        env = dict(os.environ)
        if model_settings.codex_home:
            env["CODEX_HOME"] = str(Path(model_settings.codex_home).expanduser().resolve())

        try:
            proc = subprocess.run(
                build_codex_exec_command(
                    cwd=cwd,
                    output_path=output_path,
                    model=model_settings.model,
                    schema_path=schema_path,
                    reasoning_effort=model_settings.codex_reasoning_effort,
                ),
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                env=env,
                timeout=model_settings.codex_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout = model_settings.codex_timeout_seconds or 0
            raise RuntimeError(f"Codex exec timed out after {timeout} seconds.") from exc

        usage = parse_codex_usage_from_jsonl(proc.stdout)
        final_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""

        if proc.returncode != 0 and not final_text:
            raise RuntimeError(
                f"Codex exec failed with exit code {proc.returncode}: {_failure_message(proc)}"
            )

        data = parse_last_message_json(final_text)
        return data, final_text, usage
