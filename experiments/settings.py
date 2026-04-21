from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelSettings:
    provider: str = "openai"  # "openai", "gemini", or "codex"
    model: str = "gpt-5.1"
    temperature: float = 0.2
    max_output_tokens: Optional[int] = None  # leave None to let providers decide
    base_url: Optional[str] = None  # allow overrides for local gateways
    gemini_api_key_env: Optional[str] = None  # e.g., GEMINI_API_KEY or GOOGLE_API_KEY
    codex_home: Optional[str] = None  # optional isolated CODEX_HOME for subprocess calls
    codex_reasoning_effort: Optional[str] = "medium"  # override global Codex config for experiments
    codex_timeout_seconds: Optional[int] = 300  # subprocess timeout for codex exec


@dataclass
class RunSettings:
    seeds: int = 20
    output_dir: Path = Path("runs")
    dry_run: bool = False
