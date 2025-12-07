import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    kind: str
    payload: Dict[str, Any]
    timestamp: str

    @classmethod
    def now(cls, kind: str, payload: Dict[str, Any]) -> "Event":
        return cls(kind=kind, payload=payload, timestamp=datetime.utcnow().isoformat())


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def event_log_to_jsonl(path: Path, events: List[Event]) -> None:
    write_jsonl(path, [asdict(evt) for evt in events])


def safe_float(value: Any, fallback: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def clamp_bid(bid: float, minimum: float = 0.0, maximum: float = 1e9) -> float:
    return max(minimum, min(bid, maximum))
