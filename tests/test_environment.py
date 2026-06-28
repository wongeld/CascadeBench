"""
tests/test_environment.py

Tests for core/environment.py — documents, error injection, ground truth.
"""

import pytest
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import (
    Document,
    Environment,
    ErrorInjector,
    build_milestone1_environment,
)


class TestDocument:

    def test_document_creation(self):
        doc = Document(doc_id="D1", content="Test content.", source_type="paper", reliability=0.9)
        assert doc.doc_id == "D1"
        assert doc.reliability == 0.9

    def test_document_roundtrip(self):
        doc = Document(doc_id="D1", content="Test.", source_type="blog", reliability=0.5)
        d = doc.to_dict()
        doc2 = Document.from_dict(d)
        assert doc.doc_id == doc2.doc_id
        assert doc.content == doc2.content


class TestMilestone1Environment:

    def setup_method(self):
        self.env = build_milestone1_environment()

    def test_has_5_documents(self):
        assert len(self.env.documents) == 5

    def test_doc_ids(self):
        ids = {d.doc_id for d in self.env.documents}
        assert ids == {"D1", "D2", "D3", "D4", "D5"}

    def test_ground_truth_exists(self):
        assert "primary_claim" in self.env.ground_truth

    def test_get_document(self):
        doc = self.env.get_document("D2")
        assert doc is not None
        assert doc.doc_id == "D2"

    def test_get_nonexistent_document(self):
        doc = self.env.get_document("D99")
        assert doc is None

    def test_get_documents_for_agent(self):
        docs = self.env.get_documents_for_agent("A1", max_docs=3)
        assert len(docs) <= 3
        # Should be sorted by reliability
        if len(docs) > 1:
            for i in range(len(docs) - 1):
                assert docs[i].reliability >= docs[i+1].reliability


class TestErrorInjector:

    def setup_method(self):
        self.env = build_milestone1_environment()
        self.original_d2 = copy.copy(self.env.get_document("D2"))

    def test_flip_sentence_changes_content(self):
        self.env.apply_injection(
            {"type": "flip_sentence", "doc_id": "D2", "target_sentence_idx": 0}
        )
        d2 = self.env.get_document("D2")
        assert d2.content != self.original_d2.content

    def test_flip_sentence_logged(self):
        self.env.apply_injection(
            {"type": "flip_sentence", "doc_id": "D2", "target_sentence_idx": 0}
        )
        log = self.env.get_injection_log()
        assert len(log) == 1
        assert log[0]["injection_type"] == "flip_sentence"
        assert log[0]["doc_id"] == "D2"
        assert log[0]["original"] != log[0]["corrupted"]

    def test_biased_claim_appends_text(self):
        original = self.env.get_document("D2").content
        self.env.apply_injection(
            {"type": "biased_claim", "doc_id": "D2",
             "bias_text": " But experts disagree."}
        )
        d2 = self.env.get_document("D2")
        assert "But experts disagree." in d2.content
        assert len(d2.content) > len(original)

    def test_missing_evidence_removes_sentence(self):
        d2_before = self.env.get_document("D2")
        sentence_count_before = len([s for s in d2_before.content.split(".") if s.strip()])
        self.env.apply_injection(
            {"type": "missing_evidence", "doc_id": "D2", "removal_sentence_idx": 0}
        )
        d2_after = self.env.get_document("D2")
        sentence_count_after = len([s for s in d2_after.content.split(".") if s.strip()])
        assert sentence_count_after < sentence_count_before

    def test_fabricated_social_post_sets_low_reliability(self):
        self.env.apply_injection(
            {"type": "fabricated_social_post", "doc_id": "D3",
             "handle": "@FakeExpert", "claim_text": "New study overturns everything."}
        )
        d3 = self.env.get_document("D3")
        assert d3.reliability <= 0.2
        assert "FakeExpert" in d3.content

    def test_retracted_paper_signal_sets_zero_reliability(self):
        self.env.apply_injection(
            {"type": "retracted_paper_signal", "doc_id": "D1"}
        )
        d1 = self.env.get_document("D1")
        assert d1.reliability == 0.0
        assert "RETRACTION" in d1.content

    def test_confidence_distortion(self):
        self.env.apply_injection(
            {"type": "confidence_distortion", "doc_id": "D4", "new_reliability": 0.99}
        )
        d4 = self.env.get_document("D4")
        assert d4.reliability == 0.99

    def test_unknown_doc_raises(self):
        with pytest.raises(ValueError, match="D99"):
            self.env.apply_injection(
                {"type": "flip_sentence", "doc_id": "D99"}
            )

    def test_unknown_injection_type_raises(self):
        with pytest.raises(ValueError, match="Unknown injection type"):
            self.env.apply_injection(
                {"type": "make_it_up", "doc_id": "D1"}
            )

    def test_multiple_injections_all_logged(self):
        self.env.apply_injection({"type": "flip_sentence", "doc_id": "D2"})
        self.env.apply_injection({"type": "biased_claim", "doc_id": "D3"})
        log = self.env.get_injection_log()
        assert len(log) == 2
