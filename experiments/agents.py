import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .codex_exec import run_codex_json
from .settings import ModelSettings


@dataclass
class AgentPersona:
    name: str
    orientation: str
    short_goal: str
    background: str


_PERSONAS_CACHE: Optional[Tuple[List[AgentPersona], List[AgentPersona]]] = None


def load_personas(path: Optional[Path] = None) -> Tuple[List[AgentPersona], List[AgentPersona]]:
    """
    Load personas from JSON at runtime (data/personas.json by default).
    Caches after first load to avoid repeated disk reads.
    """
    global _PERSONAS_CACHE
    if _PERSONAS_CACHE is not None:
        return _PERSONAS_CACHE
    personas_path = path or Path(__file__).resolve().parent.parent / "data" / "personas.json"
    with personas_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bidders = [
        AgentPersona(
            name=item["name"],
            orientation=item["orientation"],
            short_goal=item["short_goal"],
            background=item["background"],
        )
        for item in data.get("bidders", [])
    ]
    players = [
        AgentPersona(
            name=item["name"],
            orientation=item["orientation"],
            short_goal=item["short_goal"],
            background=item["background"],
        )
        for item in data.get("players", [])
    ]
    _PERSONAS_CACHE = (bidders, players)
    return bidders, players


BIDDER_PERSONAS, PLAYER_PERSONAS = load_personas()


def pick_personas(source: List[AgentPersona], k: int, seed: int) -> List[AgentPersona]:
    rng = random.Random(seed)
    pool = source.copy()
    rng.shuffle(pool)
    result: List[AgentPersona] = []
    while len(result) < k:
        if not pool:
            pool = source.copy()
            rng.shuffle(pool)
        result.append(pool.pop())
    return result[:k]


def extract_output_text(response: object) -> str:
    # Compatible with responses API objects where output content is nested.
    for attr in ("output", "choices"):
        if hasattr(response, attr):
            items = getattr(response, attr)
            if items:
                first = items[0]
                if hasattr(first, "content") and first.content:
                    content = first.content[0]
                    if hasattr(content, "text") and content.text:
                        return content.text
    if hasattr(response, "output_text"):
        text_val = getattr(response, "output_text")
        if isinstance(text_val, str):
            return text_val
    return ""


def adapt_persona_for_provider(persona: AgentPersona, provider: str) -> AgentPersona:
    if provider != "gemini":
        return persona
    # Gemini struggles with very long prompts; use a compact persona summary.
    compact_bg = (
        f"You are a {persona.orientation}-oriented agent. Goal: {persona.short_goal}. "
        "Stay consistent with this orientation, keep private info private, follow rules given. "
        "Be decisive, concise, and output only the required JSON schema."
    )
    return AgentPersona(
        name=persona.name,
        orientation=persona.orientation,
        short_goal=persona.short_goal,
        background=compact_bg,
    )


def call_agent_json(
    client: Optional[OpenAI],
    model_settings: ModelSettings,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Dict] = None,
    dry_run: bool = False,
    metadata_sink: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, str]:
    if dry_run:
        if metadata_sink is not None:
            metadata_sink.update({"provider": model_settings.provider, "usage": None})
        return {}, '{"stub": true}'
    usage = None
    if model_settings.provider == "codex":
        data, text, usage = run_codex_json(
            prompt=_build_codex_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=response_schema,
            ),
            model_settings=model_settings,
            response_schema=response_schema,
        )
    elif model_settings.provider == "gemini":
        data, text = _call_gemini_json(
            model_settings=model_settings,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
        )
    else:
        if client is None:
            client = openai_client_from_settings(model_settings)
        data, text = _call_openai_json(
            client=client,
            model_settings=model_settings,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if metadata_sink is not None:
        metadata_sink.update({"provider": model_settings.provider, "usage": usage})
    return data, text


def _build_codex_prompt(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "You are producing one decision for a structured market simulation.",
        "Return exactly one JSON object and nothing else.",
        "",
        "System instructions:",
        system_prompt.strip(),
        "",
        "User state:",
        user_prompt.strip(),
    ]
    if isinstance(response_schema, dict):
        properties = response_schema.get("properties")
        if isinstance(properties, dict) and properties:
            lines.extend(["", "Required top-level JSON fields:"])
            for key, value in properties.items():
                schema_type = ""
                if isinstance(value, dict):
                    schema_type = str(value.get("type") or "").strip().lower()
                if schema_type:
                    lines.append(f'- "{key}": {schema_type}')
                else:
                    lines.append(f'- "{key}"')
    return "\n".join(lines).strip() + "\n"


def openai_client_from_settings(settings: ModelSettings) -> OpenAI:
    if settings.base_url:
        return OpenAI(base_url=settings.base_url)
    return OpenAI()


def _call_openai_json(
    client: OpenAI,
    model_settings: ModelSettings,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Dict, str]:
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]
    last_error = None
    for attempt in range(3):
        try:
            kwargs = {
                "model": model_settings.model,
                "input": messages,
                "temperature": model_settings.temperature,
            }
            response = client.responses.create(**kwargs)
            break
        except Exception as exc:  # broad catch to retry transient issues
            last_error = exc
            if attempt == 2:
                raise
            sleep_for = 1.5 * (attempt + 1)
            time.sleep(sleep_for)
    else:
        raise last_error if last_error else RuntimeError("Unknown error calling model.")
    text = extract_output_text(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    return data, text


def _call_gemini_json(
    model_settings: ModelSettings,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Dict] = None,
) -> Tuple[Dict, str]:
    key = (
        os.getenv(model_settings.gemini_api_key_env or "GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY for Gemini provider.")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-genai is not installed. Run pip install google-genai.") from exc

    client = genai.Client(api_key=key)
    combined = f"System instructions:\n{system_prompt}\n\nUser request:\n{user_prompt}\n\nReturn strictly JSON."
    last_error = None
    text = ""
    for attempt in range(3):
        try:
            config = {
                "temperature": model_settings.temperature,
                "response_mime_type": "application/json",
                **({"response_schema": response_schema} if response_schema else {}),
            }
            resp = client.models.generate_content(
                model=model_settings.model,
                contents=[{"role": "user", "parts": [{"text": combined}]}],
                config=config,
            )
            text = getattr(resp, "text", "") or ""
            if not text:
                for cand in getattr(resp, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts is None:
                        continue
                    for part in parts:
                        if getattr(part, "text", None):
                            text = part.text
                            break
                    if text:
                        break
            break
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise
            sleep_for = 1.5 * (attempt + 1)
            time.sleep(sleep_for)
    else:
        raise last_error if last_error else RuntimeError("Unknown error calling model.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    # If schema was provided and no data returned, raise to trigger retry/fail fast.
    if response_schema and (not isinstance(data, dict) or not data):
        raise ValueError("Gemini returned empty/invalid JSON.")
    return data, text
