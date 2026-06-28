"""
logging/trace_store.py

Append-only JSONL trace store.
Every experiment event is written to:
  logs/<experiment_id>/trace.jsonl

Supports:
  - Append: append_event(entry)
  - Replay: load_trace(experiment_id) → List[dict]
  - Filtering: filter_events(experiment_id, agent=None, event_type=None, round=None)

The store is the ground truth for post-hoc analysis.
Do NOT modify past entries — only append.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


LOGS_ROOT = Path("logs")


class TraceStore:
    """
    Append-only JSONL file per experiment.
    Thread-safe: uses file append mode per write.
    """

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self._dir = LOGS_ROOT / experiment_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self._dir / "trace.jsonl"

    def append_event(self, entry: dict) -> None:
        """Append one log entry to the JSONL trace file."""
        # Ensure mandatory fields
        entry.setdefault("experiment_id", self.experiment_id)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(self._trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_trace(self) -> List[dict]:
        """Load and return all log entries from the JSONL file."""
        if not self._trace_file.exists():
            return []
        entries = []
        with open(self._trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def filter_events(
        self,
        agent: Optional[str] = None,
        event_type: Optional[str] = None,
        round_num: Optional[int] = None,
    ) -> List[dict]:
        """Filter trace entries by agent, event_type, and/or round."""
        entries = self.load_trace()
        result = []
        for e in entries:
            if agent and e.get("agent") != agent:
                continue
            if event_type and e.get("event_type") != event_type:
                continue
            if round_num is not None and e.get("round") != round_num:
                continue
            result.append(e)
        return result

    def write_json(self, filename: str, data) -> None:
        """Write a JSON file to the experiment log directory."""
        path = self._dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_json(self, filename: str) -> Optional[dict]:
        """Read a JSON file from the experiment log directory."""
        path = self._dir / filename
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def log_dir(self) -> Path:
        return self._dir

    def __repr__(self) -> str:
        n = len(self.load_trace())
        return f"TraceStore(experiment_id={self.experiment_id!r}, entries={n})"
