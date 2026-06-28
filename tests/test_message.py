"""
tests/test_message.py

Tests for core/message.py — schema validation and edge cases.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError
from core.message import Message


def valid_message_dict():
    return {
        "claim": "Vitamin D reduces infection risk.",
        "confidence": 0.85,
        "evidence_ids": ["D1", "D2"],
        "reasoning": "Meta-analysis supports protective effect.",
        "type": "support",
        "source_agent": "A1",
        "round": 1,
    }


class TestMessageValidation:

    def test_valid_message_creates_ok(self):
        msg = Message(**valid_message_dict())
        assert msg.claim == "Vitamin D reduces infection risk."
        assert msg.confidence == 0.85
        assert msg.type == "support"

    def test_confidence_out_of_range_high(self):
        d = valid_message_dict()
        d["confidence"] = 1.5
        with pytest.raises(ValidationError):
            Message(**d)

    def test_confidence_out_of_range_low(self):
        d = valid_message_dict()
        d["confidence"] = -0.1
        with pytest.raises(ValidationError):
            Message(**d)

    def test_invalid_type(self):
        d = valid_message_dict()
        d["type"] = "agree"  # not a valid type
        with pytest.raises(ValidationError):
            Message(**d)

    def test_empty_claim_rejected(self):
        d = valid_message_dict()
        d["claim"] = "   "
        with pytest.raises(ValidationError):
            Message(**d)

    def test_empty_reasoning_rejected(self):
        d = valid_message_dict()
        d["reasoning"] = ""
        with pytest.raises(ValidationError):
            Message(**d)

    def test_all_valid_types(self):
        for t in ["support", "contradict", "question"]:
            d = valid_message_dict()
            d["type"] = t
            msg = Message(**d)
            assert msg.type == t

    def test_roundtrip_serialization(self):
        msg = Message(**valid_message_dict())
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg.claim == msg2.claim
        assert msg.confidence == msg2.confidence
        assert msg.type == msg2.type

    def test_empty_evidence_ids_allowed(self):
        d = valid_message_dict()
        d["evidence_ids"] = []
        msg = Message(**d)
        assert msg.evidence_ids == []

    def test_confidence_boundaries(self):
        d = valid_message_dict()
        d["confidence"] = 0.0
        msg = Message(**d)
        assert msg.confidence == 0.0

        d["confidence"] = 1.0
        msg = Message(**d)
        assert msg.confidence == 1.0

    def test_repr_contains_key_fields(self):
        msg = Message(**valid_message_dict())
        r = repr(msg)
        assert "A1" in r
        assert "support" in r
