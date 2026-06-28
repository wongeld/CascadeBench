"""
tests/test_metrics.py

Tests for metrics/cascade_metrics.py using synthetic trace data.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.cascade_metrics import (
    false_consensus_rate,
    cascade_depth,
    time_to_consensus,
    error_amplification_factor,
    influence_scores,
    belief_stability,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "primary_claim": "Vitamin D supplementation reduces the risk of acute respiratory infections.",
    "Does vitamin D supplementation reduce respiratory infection risk?": "yes",
}

TARGET_CLAIM = "Vitamin D supplementation reduces the risk of acute respiratory infections."

FALSE_CLAIM = "Vitamin D supplementation does not reduce the risk of acute respiratory infections."

AGENT_IDS = ["A1", "A2", "A3"]


def make_belief_update_event(agent, claim, prior, new, caused_msg=None, round_num=1):
    return {
        "event_type": "belief_update",
        "round": round_num,
        "agent": agent,
        "content": {
            "claim": claim,
            "prior_confidence": prior,
            "new_confidence": new,
            "delta": new - prior,
            "caused_by_message": caused_msg,
            "caused_by_doc": None,
        },
    }


def make_round_end_event(round_num, snapshot):
    return {
        "event_type": "round_end",
        "round": round_num,
        "agent": "system",
        "content": {"belief_snapshot": snapshot},
    }


def make_message_sent_event(agent, to, claim, confidence, evidence_ids, round_num=1):
    return {
        "event_type": "message_sent",
        "round": round_num,
        "agent": agent,
        "content": {
            "to": to,
            "message": {
                "claim": claim,
                "confidence": confidence,
                "evidence_ids": evidence_ids,
                "reasoning": "test",
                "type": "support",
                "source_agent": agent,
                "round": round_num,
            },
        },
    }


class TestFalseConsensusRate:

    def test_all_agents_false(self):
        """All agents believe false claim → FCR = 1.0"""
        final_beliefs = {
            aid: {FALSE_CLAIM: 0.8}
            for aid in AGENT_IDS
        }
        fcr = false_consensus_rate(final_beliefs, GROUND_TRUTH, FALSE_CLAIM)
        assert fcr == 1.0

    def test_no_agents_false(self):
        """No agents believe false claim → FCR = 0.0"""
        final_beliefs = {
            aid: {TARGET_CLAIM: 0.9}  # true claim only
            for aid in AGENT_IDS
        }
        fcr = false_consensus_rate(final_beliefs, GROUND_TRUTH, FALSE_CLAIM)
        assert fcr == 0.0

    def test_partial_false(self):
        """1 of 3 agents believes false claim → FCR ≈ 0.33"""
        final_beliefs = {
            "A1": {FALSE_CLAIM: 0.8},
            "A2": {TARGET_CLAIM: 0.9},
            "A3": {TARGET_CLAIM: 0.85},
        }
        fcr = false_consensus_rate(final_beliefs, GROUND_TRUTH, FALSE_CLAIM)
        assert 0.3 < fcr <= 0.4

    def test_below_threshold_not_counted(self):
        """Belief below threshold is not counted as false consensus."""
        final_beliefs = {
            aid: {FALSE_CLAIM: 0.3}
            for aid in AGENT_IDS
        }
        fcr = false_consensus_rate(
            final_beliefs, GROUND_TRUTH, FALSE_CLAIM, confidence_threshold=0.5
        )
        assert fcr == 0.0


class TestCascadeDepth:

    def test_no_injections_zero_depth(self):
        trace = [make_belief_update_event("A1", TARGET_CLAIM, 0.5, 0.7)]
        depth = cascade_depth(trace=trace, injection_log=[])
        assert depth == 0

    def test_injection_causes_belief_update(self):
        """Belief update referencing an injected doc → depth ≥ 1"""
        caused_msg = {"source_agent": "A2", "confidence": 0.8, "evidence_ids": ["D2"], "round": 1}
        trace = [
            make_belief_update_event("A1", FALSE_CLAIM, 0.5, 0.8, caused_msg=caused_msg, round_num=1)
        ]
        injection_log = [{"injection_type": "flip_sentence", "doc_id": "D2", "timestep": 0}]
        depth = cascade_depth(trace=trace, injection_log=injection_log)
        assert depth >= 1


class TestTimeToConsensus:

    def test_consensus_reached_round_2(self):
        trace = [
            make_round_end_event(1, {
                "A1": {TARGET_CLAIM: 0.6},
                "A2": {TARGET_CLAIM: 0.4},
                "A3": {TARGET_CLAIM: 0.3},
            }),
            make_round_end_event(2, {
                "A1": {TARGET_CLAIM: 0.8},
                "A2": {TARGET_CLAIM: 0.7},
                "A3": {TARGET_CLAIM: 0.75},
            }),
        ]
        ttc = time_to_consensus(trace, TARGET_CLAIM, AGENT_IDS)
        assert ttc == 2

    def test_no_consensus_returns_none(self):
        trace = [
            make_round_end_event(1, {
                "A1": {TARGET_CLAIM: 0.3},
                "A2": {TARGET_CLAIM: 0.2},
                "A3": {TARGET_CLAIM: 0.1},
            }),
        ]
        ttc = time_to_consensus(trace, TARGET_CLAIM, AGENT_IDS)
        assert ttc is None


class TestErrorAmplificationFactor:

    def test_eaf_above_one_means_amplified(self):
        """If mean final false confidence > initial, EAF > 1."""
        final_beliefs = {
            "A1": {FALSE_CLAIM: 0.9},
            "A2": {FALSE_CLAIM: 0.8},
        }
        eaf = error_amplification_factor(
            final_beliefs, GROUND_TRUTH, FALSE_CLAIM, [], initial_confidence=0.5
        )
        assert eaf > 1.0

    def test_eaf_zero_when_no_false_beliefs(self):
        final_beliefs = {
            "A1": {TARGET_CLAIM: 0.9},
        }
        eaf = error_amplification_factor(
            final_beliefs, GROUND_TRUTH, FALSE_CLAIM, [], initial_confidence=0.5
        )
        assert eaf == 0.0


class TestInfluenceScores:

    def test_returns_empty_for_empty_trace(self):
        scores = influence_scores([])
        assert scores == {}

    def test_agents_with_more_causal_messages_get_higher_scores(self):
        """A1 causes many updates → should have higher PageRank than A3."""
        msg_a1 = {"source_agent": "A1", "confidence": 0.8, "evidence_ids": [], "round": 1}
        msg_a3 = {"source_agent": "A3", "confidence": 0.7, "evidence_ids": [], "round": 1}
        trace = (
            [make_belief_update_event("A2", TARGET_CLAIM, 0.5, 0.7, caused_msg=msg_a1, round_num=r)
             for r in range(1, 6)]
            + [make_belief_update_event("A1", FALSE_CLAIM, 0.5, 0.6, caused_msg=msg_a3, round_num=1)]
        )
        scores = influence_scores(trace)
        if "A1" in scores and "A3" in scores:
            assert scores["A1"] >= scores["A3"]


class TestBeliefStability:

    def test_no_updates_means_fully_stable(self):
        stability = belief_stability(trace=[], agent_ids=AGENT_IDS)
        for aid in AGENT_IDS:
            assert stability[aid] == 1.0

    def test_high_deltas_means_low_stability(self):
        trace = [
            make_belief_update_event("A1", TARGET_CLAIM, 0.1, 0.9, round_num=r)
            for r in range(1, 6)
        ]
        stability = belief_stability(trace=trace, agent_ids=AGENT_IDS)
        assert stability["A1"] < 0.5  # large deltas = low stability
        assert stability["A2"] == 1.0  # no updates = stable


class TestComputeAllMetrics:

    def test_returns_all_required_keys(self):
        required_keys = [
            "false_consensus_rate",
            "cascade_depth",
            "time_to_consensus",
            "error_amplification_factor",
            "influence_scores",
            "belief_stability",
            "num_belief_updates",
            "total_messages",
            "num_injections",
        ]
        final_beliefs = {aid: {TARGET_CLAIM: 0.7} for aid in AGENT_IDS}
        metrics = compute_all_metrics(
            trace=[],
            final_beliefs=final_beliefs,
            ground_truth=GROUND_TRUTH,
            target_claim=TARGET_CLAIM,
            injection_log=[],
            num_rounds=5,
            agent_ids=AGENT_IDS,
        )
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
