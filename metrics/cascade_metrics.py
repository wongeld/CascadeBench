"""
metrics/cascade_metrics.py

Post-hoc analysis metrics for error propagation experiments.

Metrics implemented:
  1. False Consensus Rate (FCR)
     % agents whose final belief confidence on the injected false claim > threshold

  2. Cascade Depth
     Longest chain: injection → message_sent → belief_update → message_sent → ...
     Traced from the JSONL trace via causal message references.

  3. Time-to-Consensus (TTC)
     First round where ≥ consensus_threshold fraction of agents agree on any claim

  4. Error Amplification Factor (EAF)
     mean(final agent confidence on false claim) / initial injection confidence
     Measures whether the network amplified the error

  5. Influence Score
     PageRank on the causal message graph
     Nodes = agents, edges = "A sent message that caused B's belief update"

All metrics operate on:
  - trace: List[dict] loaded from trace.jsonl
  - final_beliefs: {agent_id: {claim: confidence}}
  - ground_truth: {question: answer}
  - target_claim: str (the claim being tracked)
  - injection_log: list of injection records
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Helper: is a claim "false" according to ground truth?
# ---------------------------------------------------------------------------

def _is_claim_false(claim: str, ground_truth: dict) -> bool:
    """
    Check if a claim contradicts ground truth.
    Simple heuristic: if ground truth says "yes" and claim contains "does not",
    "no", "not", "decreases", etc. it's treated as false.
    """
    primary_gt = ground_truth.get("primary_claim", "")
    if not primary_gt:
        return False

    gt_words = set(primary_gt.lower().split())
    claim_words = set(claim.lower().split())

    negation_markers = {"not", "does not", "cannot", "doesn't", "no", "false",
                        "incorrect", "decreases", "reduces not", "disproves"}

    # If claim has negation markers that the ground truth doesn't, it's likely false
    claim_lower = claim.lower()
    for marker in negation_markers:
        if marker in claim_lower and marker not in primary_gt.lower():
            return True

    return False


def _claim_matches_target(claim: str, target_claim: str) -> bool:
    """Check if a claim string relates to the target claim."""
    if not claim or not target_claim:
        return False
    # Exact match
    if claim.strip() == target_claim.strip():
        return True
    # Keyword overlap
    c_words = set(claim.lower().split()) - {"the", "a", "an", "is", "are", "of", "to", "and"}
    t_words = set(target_claim.lower().split()) - {"the", "a", "an", "is", "are", "of", "to", "and"}
    if not t_words:
        return False
    overlap = c_words & t_words
    return len(overlap) / len(t_words) > 0.4


# ---------------------------------------------------------------------------
# 1. False Consensus Rate
# ---------------------------------------------------------------------------

def false_consensus_rate(
    final_beliefs: Dict[str, Dict[str, float]],
    ground_truth: dict,
    target_claim: str,
    confidence_threshold: float = 0.5,
) -> float:
    """
    % of agents with confidence > threshold on the injected false claim.

    The "false claim" is any belief the agent holds that contradicts ground truth
    AND relates to the target claim.

    Returns:
        float in [0.0, 1.0]
    """
    if not final_beliefs:
        return 0.0

    false_believers = 0
    total = len(final_beliefs)

    for agent_id, beliefs in final_beliefs.items():
        for claim, confidence in beliefs.items():
            if confidence > confidence_threshold and _claim_matches_target(claim, target_claim):
                if _is_claim_false(claim, ground_truth):
                    false_believers += 1
                    break  # count agent once

    return false_believers / total


# ---------------------------------------------------------------------------
# 2. Cascade Depth
# ---------------------------------------------------------------------------

def cascade_depth(trace: List[dict], injection_log: List[dict]) -> int:
    """
    Longest causal chain from an injected error through belief updates.

    Algorithm:
      1. Find all belief_update events caused by a message that traces back to an injection
      2. Build a causal graph: belief_update → message_sent → belief_update → ...
      3. Find longest path in this DAG

    Simplified: counts the number of unique belief_update events whose
    causal chain can be traced back to an injection event.
    Returns length of longest such chain.
    """
    if not injection_log:
        return 0

    # Find injected doc IDs
    injected_docs = {r.get("doc_id") for r in injection_log}

    # Find all belief_update events referencing injected docs
    # Build message propagation chain
    belief_updates = [e for e in trace if e.get("event_type") == "belief_update"]
    messages_sent = [e for e in trace if e.get("event_type") == "message_sent"]

    # Map: (source_agent, round) → evidence_ids used
    sent_evidence: Dict = {}
    for e in messages_sent:
        msg = e.get("content", {}).get("message", {})
        key = (e.get("agent"), e.get("round"))
        sent_evidence[key] = msg.get("evidence_ids", [])

    # For each belief_update, check if caused_by_doc is an injected doc
    # or if a message from a previous round that referenced injected docs caused it
    def is_tainted(update_event: dict, depth: int = 0) -> int:
        """Return cascade depth from this update event."""
        if depth > 20:  # circuit breaker
            return depth
        content = update_event.get("content", {})
        caused_doc = content.get("caused_by_doc")
        caused_msg = content.get("caused_by_message", {}) or {}
        caused_ev_ids = caused_msg.get("evidence_ids", [])

        # Direct injection contact
        if caused_doc and caused_doc in injected_docs:
            return 1
        if any(eid in injected_docs for eid in caused_ev_ids):
            return 1

        # Trace through source message
        source_agent = caused_msg.get("source_agent")
        source_round = caused_msg.get("round", 0)
        if source_agent and source_round > 0:
            prev_key = (source_agent, source_round)
            prev_ev = sent_evidence.get(prev_key, [])
            if any(eid in injected_docs for eid in prev_ev):
                return 2
            # Could recurse further but keeping simple for M1
        return 0

    depths = [is_tainted(u) for u in belief_updates]
    return max(depths) if depths else 0


# ---------------------------------------------------------------------------
# 3. Time-to-Consensus
# ---------------------------------------------------------------------------

def time_to_consensus(
    trace: List[dict],
    target_claim: str,
    agent_ids: List[str],
    consensus_threshold: float = 0.67,
    confidence_threshold: float = 0.5,
) -> Optional[int]:
    """
    First round where ≥ consensus_threshold fraction of agents agree on target_claim.

    Returns round number (1-indexed), or None if consensus never reached.
    """
    # Collect belief snapshots per round from round_end events
    round_ends = [e for e in trace if e.get("event_type") == "round_end"]

    for event in round_ends:
        round_num = event.get("round", 0)
        snapshot = event.get("content", {}).get("belief_snapshot", {})
        if not snapshot:
            continue

        agreeing = 0
        for aid in agent_ids:
            agent_beliefs = snapshot.get(aid, {})
            for claim, conf in agent_beliefs.items():
                if _claim_matches_target(claim, target_claim) and conf > confidence_threshold:
                    agreeing += 1
                    break

        if len(agent_ids) > 0 and agreeing / len(agent_ids) >= consensus_threshold:
            return round_num

    return None


# ---------------------------------------------------------------------------
# 4. Error Amplification Factor
# ---------------------------------------------------------------------------

def error_amplification_factor(
    final_beliefs: Dict[str, Dict[str, float]],
    ground_truth: dict,
    target_claim: str,
    injection_log: List[dict],
    initial_confidence: float = 0.5,
) -> float:
    """
    EAF = mean(final false belief confidence) / initial_confidence

    > 1.0 means the network amplified the error.
    < 1.0 means the network suppressed/corrected it.
    """
    false_confidences = []
    for agent_id, beliefs in final_beliefs.items():
        for claim, confidence in beliefs.items():
            if _claim_matches_target(claim, target_claim) and _is_claim_false(claim, ground_truth):
                false_confidences.append(confidence)

    if not false_confidences:
        return 0.0

    mean_false_conf = sum(false_confidences) / len(false_confidences)
    return round(mean_false_conf / initial_confidence, 4) if initial_confidence > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. Influence Score (PageRank on causal message graph)
# ---------------------------------------------------------------------------

def influence_scores(trace: List[dict]) -> Dict[str, float]:
    """
    Compute PageRank influence scores from causal message graph.

    Nodes = agents
    Edges = (source_agent → receiving_agent) with weight = confidence

    An agent is influential if its messages caused many belief updates in others.
    """
    # Build directed graph from belief_update events with causal message attribution
    G = nx.DiGraph()

    belief_updates = [e for e in trace if e.get("event_type") == "belief_update"]
    for event in belief_updates:
        receiver = event.get("agent")
        content = event.get("content", {})
        caused_msg = content.get("caused_by_message", {})
        if caused_msg:
            sender = caused_msg.get("source_agent")
            confidence = caused_msg.get("confidence", 0.5)
            if sender and receiver and sender != receiver:
                if G.has_edge(sender, receiver):
                    G[sender][receiver]["weight"] += confidence
                else:
                    G.add_edge(sender, receiver, weight=confidence)

    if G.number_of_nodes() == 0:
        return {}

    try:
        scores = nx.pagerank(G, weight="weight", alpha=0.85)
        return {k: round(v, 6) for k, v in scores.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 6. Belief Stability (bonus metric)
# ---------------------------------------------------------------------------

def belief_stability(trace: List[dict], agent_ids: List[str]) -> Dict[str, float]:
    """
    For each agent, measure how much their beliefs fluctuated.
    Stability = 1 - mean(|delta| across all belief updates)
    Higher = more stable.
    """
    deltas: Dict[str, List[float]] = defaultdict(list)

    for event in trace:
        if event.get("event_type") == "belief_update":
            aid = event.get("agent")
            delta = abs(event.get("content", {}).get("delta", 0.0))
            if aid:
                deltas[aid].append(delta)

    result = {}
    for aid in agent_ids:
        agent_deltas = deltas.get(aid, [])
        if agent_deltas:
            result[aid] = round(1.0 - (sum(agent_deltas) / len(agent_deltas)), 4)
        else:
            result[aid] = 1.0  # no updates = fully stable

    return result


# ---------------------------------------------------------------------------
# Composite: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    trace: List[dict],
    final_beliefs: Dict[str, Dict[str, float]],
    ground_truth: dict,
    target_claim: str,
    injection_log: List[dict],
    num_rounds: int,
    agent_ids: List[str],
    fcr_confidence_threshold: float = 0.5,
    consensus_threshold: float = 0.67,
    initial_injection_confidence: float = 0.5,
) -> dict:
    """
    Compute and return all metrics as a single dictionary.
    """
    fcr = false_consensus_rate(
        final_beliefs=final_beliefs,
        ground_truth=ground_truth,
        target_claim=target_claim,
        confidence_threshold=fcr_confidence_threshold,
    )

    cd = cascade_depth(trace=trace, injection_log=injection_log)

    ttc = time_to_consensus(
        trace=trace,
        target_claim=target_claim,
        agent_ids=agent_ids,
        consensus_threshold=consensus_threshold,
        confidence_threshold=fcr_confidence_threshold,
    )

    eaf = error_amplification_factor(
        final_beliefs=final_beliefs,
        ground_truth=ground_truth,
        target_claim=target_claim,
        injection_log=injection_log,
        initial_confidence=initial_injection_confidence,
    )

    influence = influence_scores(trace=trace)
    stability = belief_stability(trace=trace, agent_ids=agent_ids)

    return {
        "false_consensus_rate": round(fcr, 4),
        "cascade_depth": cd,
        "time_to_consensus": ttc,
        "error_amplification_factor": eaf,
        "influence_scores": influence,
        "belief_stability": stability,
        "num_belief_updates": sum(
            1 for e in trace if e.get("event_type") == "belief_update"
        ),
        "total_messages": sum(
            1 for e in trace if e.get("event_type") == "message_sent"
        ),
        "num_injections": len(injection_log),
    }
