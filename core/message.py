"""
core/message.py

Strict structured message schema for all inter-agent communication.
No free-form text — every message is validated JSON.

Schema:
{
  "claim": str,
  "confidence": float (0.0–1.0),
  "evidence_ids": List[str],
  "reasoning": str,
  "type": "support" | "contradict" | "question",
  "source_agent": str,
  "round": int
}
"""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


MessageType = Literal["support", "contradict", "question"]


class Message(BaseModel):
    """A single structured inter-agent message."""

    claim: str = Field(..., description="The claim being communicated.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the claim (0–1)."
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of documents/sentences supporting this message.",
    )
    reasoning: str = Field(
        ..., description="Short natural-language explanation of the claim."
    )
    type: MessageType = Field(
        ..., description="Relationship to the claim: support, contradict, or question."
    )
    source_agent: str = Field(..., description="ID of the sending agent.")
    round: int = Field(..., ge=0, description="Simulation round number.")

    @field_validator("claim")
    @classmethod
    def claim_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim must not be empty")
        return v.strip()

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning must not be empty")
        return v.strip()

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"Message(from={self.source_agent}, round={self.round}, "
            f"type={self.type}, conf={self.confidence:.2f}, "
            f"claim={self.claim[:60]!r})"
        )
