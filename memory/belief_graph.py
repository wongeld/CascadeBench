"""
memory/belief_graph.py

BeliefGraph: directed graph where nodes are claims and edges are
supports/contradicts relationships.

Node attributes:
  - confidence: float (current agent belief, 0–1)
  - first_seen_round: int
  - last_updated_round: int

Edge attributes:
  - type: "supports" | "contradicts"
  - weight: float (confidence of the source message)
  - source_agent: str
  - round: int
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx


class BeliefGraph:
    """
    Directed belief graph for a single agent.

    Nodes  = claim strings
    Edges  = (claim_a, claim_b, type=supports|contradicts)

    Provides utilities for:
      - Adding/updating claims and relationships
      - Querying current confidence
      - Detecting contradiction clusters
      - Serialization
    """

    def __init__(self):
        self._G: nx.DiGraph = nx.DiGraph()

    # ---- Mutation -------------------------------------------------------

    def upsert_claim(
        self,
        claim: str,
        confidence: float,
        round_num: int,
    ) -> None:
        """Add or update a claim node with its current confidence."""
        if self._G.has_node(claim):
            self._G.nodes[claim]["confidence"] = confidence
            self._G.nodes[claim]["last_updated_round"] = round_num
        else:
            self._G.add_node(
                claim,
                confidence=confidence,
                first_seen_round=round_num,
                last_updated_round=round_num,
            )

    def add_relationship(
        self,
        claim_a: str,
        claim_b: str,
        rel_type: str,  # "supports" | "contradicts"
        weight: float,
        source_agent: str,
        round_num: int,
    ) -> None:
        """
        Add or strengthen a directed relationship edge.
        If the edge exists, update weight and round.
        """
        if not self._G.has_node(claim_a):
            self.upsert_claim(claim_a, confidence=0.5, round_num=round_num)
        if not self._G.has_node(claim_b):
            self.upsert_claim(claim_b, confidence=0.5, round_num=round_num)

        self._G.add_edge(
            claim_a,
            claim_b,
            type=rel_type,
            weight=weight,
            source_agent=source_agent,
            round=round_num,
        )

    # ---- Queries --------------------------------------------------------

    def get_confidence(self, claim: str) -> Optional[float]:
        """Return current confidence for a claim, or None if unknown."""
        if self._G.has_node(claim):
            return self._G.nodes[claim].get("confidence")
        return None

    def get_supporters(self, claim: str) -> List[Tuple[str, dict]]:
        """Return claims that support this claim."""
        return [
            (src, data)
            for src, tgt, data in self._G.in_edges(claim, data=True)
            if data.get("type") == "supports"
        ]

    def get_contradictors(self, claim: str) -> List[Tuple[str, dict]]:
        """Return claims that contradict this claim."""
        return [
            (src, data)
            for src, tgt, data in self._G.in_edges(claim, data=True)
            if data.get("type") == "contradicts"
        ]

    def all_claims(self) -> Dict[str, float]:
        """Return {claim: confidence} for all known claims."""
        return {
            n: d.get("confidence", 0.5)
            for n, d in self._G.nodes(data=True)
        }

    def node_count(self) -> int:
        return self._G.number_of_nodes()

    def edge_count(self) -> int:
        return self._G.number_of_edges()

    # ---- Serialization --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"claim": n, **d}
                for n, d in self._G.nodes(data=True)
            ],
            "edges": [
                {"from": u, "to": v, **d}
                for u, v, d in self._G.edges(data=True)
            ],
        }

    def __repr__(self) -> str:
        return (
            f"BeliefGraph(nodes={self.node_count()}, edges={self.edge_count()})"
        )
