"""
logging/logger.py

Structured experiment event logger.
Wraps TraceStore and provides typed event-logging methods.

Event types:
  message_sent       — agent emits a message to neighbor
  message_received   — agent receives a message from neighbor
  belief_update      — agent updates a belief with causal trace
  reasoning_step     — agent completes a full reasoning step
  injection          — error injection applied to environment
  round_start        — simulation round begins
  round_end          — simulation round ends
  experiment_start   — experiment begins
  experiment_end     — experiment ends
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sim_logging.trace_store import TraceStore


class ExperimentLogger:
    """
    Typed logger for simulation events.
    All events are written to the TraceStore (JSONL).
    Also prints summary lines to console via rich if available.
    """

    def __init__(self, experiment_id: str, verbose: bool = True):
        self.experiment_id = experiment_id
        self.verbose = verbose
        self.store = TraceStore(experiment_id)
        self._use_rich = self._check_rich()

    def _check_rich(self) -> bool:
        try:
            import rich  # noqa: F401
            return True
        except ImportError:
            return False

    def _log(self, entry: dict) -> None:
        entry["experiment_id"] = self.experiment_id
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.store.append_event(entry)
        if self.verbose:
            self._print_entry(entry)

    def _print_entry(self, entry: dict) -> None:
        etype = entry.get("event_type", "?")
        rnd = entry.get("round", "-")
        agent = entry.get("agent", "-")
        if self._use_rich:
            from rich import print as rprint
            color = {
                "message_sent": "cyan",
                "message_received": "green",
                "belief_update": "yellow",
                "injection": "bold red",
                "round_start": "bold blue",
                "round_end": "blue",
                "experiment_start": "bold magenta",
                "experiment_end": "bold magenta",
                "reasoning_step": "dim",
            }.get(etype, "white")
            rprint(f"[{color}][R{rnd}][{agent}] {etype}[/{color}]", end=" ")
            content = entry.get("content", {})
            if isinstance(content, dict):
                summary = {
                    k: v for k, v in content.items()
                    if k in ("claim", "confidence", "type", "from", "to",
                             "delta", "injection_type", "doc_id")
                }
                if summary:
                    rprint(summary)
                else:
                    rprint()
            else:
                rprint(str(content)[:120])
        else:
            content = entry.get("content", {})
            print(f"[R{rnd}][{agent}] {etype} | {str(content)[:100]}")

    # ---- Typed event methods ----------------------------------------

    def log_experiment_start(self, config: dict) -> None:
        self._log({
            "event_type": "experiment_start",
            "round": 0,
            "agent": "system",
            "content": {"config": config},
        })

    def log_experiment_end(self, summary: dict) -> None:
        self._log({
            "event_type": "experiment_end",
            "round": summary.get("total_rounds", -1),
            "agent": "system",
            "content": summary,
        })

    def log_round_start(self, round_num: int) -> None:
        self._log({
            "event_type": "round_start",
            "round": round_num,
            "agent": "system",
            "content": {"round": round_num},
        })

    def log_round_end(self, round_num: int, belief_snapshot: dict) -> None:
        self._log({
            "event_type": "round_end",
            "round": round_num,
            "agent": "system",
            "content": {"belief_snapshot": belief_snapshot},
        })

    def log_message_sent(
        self,
        round_num: int,
        source_agent: str,
        target_agent: str,
        message_dict: dict,
    ) -> None:
        self._log({
            "event_type": "message_sent",
            "round": round_num,
            "agent": source_agent,
            "content": {
                "to": target_agent,
                "message": message_dict,
            },
        })

    def log_message_received(
        self,
        round_num: int,
        receiving_agent: str,
        source_agent: str,
        message_dict: dict,
    ) -> None:
        self._log({
            "event_type": "message_received",
            "round": round_num,
            "agent": receiving_agent,
            "content": {
                "from": source_agent,
                "message": message_dict,
            },
        })

    def log_belief_update(
        self,
        round_num: int,
        agent_id: str,
        claim: str,
        prior_confidence: float,
        new_confidence: float,
        caused_by_message: Optional[dict] = None,
        caused_by_doc: Optional[str] = None,
    ) -> None:
        """
        Log a belief change with full causal trace.
        Tracks: what changed, by how much, what caused it.
        """
        self._log({
            "event_type": "belief_update",
            "round": round_num,
            "agent": agent_id,
            "content": {
                "claim": claim,
                "prior_confidence": round(prior_confidence, 4),
                "new_confidence": round(new_confidence, 4),
                "delta": round(new_confidence - prior_confidence, 4),
                "caused_by_message": caused_by_message,
                "caused_by_doc": caused_by_doc,
            },
        })

    def log_reasoning_step(
        self,
        round_num: int,
        agent_id: str,
        prompt_summary: str,
        output_message: dict,
        retrieved_memory_count: int,
        docs_used: list,
    ) -> None:
        self._log({
            "event_type": "reasoning_step",
            "round": round_num,
            "agent": agent_id,
            "content": {
                "prompt_summary": prompt_summary,
                "output_message": output_message,
                "retrieved_memory_count": retrieved_memory_count,
                "docs_used": docs_used,
            },
        })

    def log_injection(self, injection_record: dict) -> None:
        self._log({
            "event_type": "injection",
            "round": injection_record.get("timestep", 0),
            "agent": "environment",
            "content": injection_record,
        })

    def get_store(self) -> TraceStore:
        return self.store
