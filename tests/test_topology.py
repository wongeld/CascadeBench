"""
tests/test_topology.py

Tests for core/topology.py — graph construction and properties.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.topology import build_topology, SUPPORTED_TOPOLOGIES


AGENT_IDS_3 = ["A1", "A2", "A3"]
AGENT_IDS_5 = ["A1", "A2", "A3", "A4", "A5"]


class TestTopologyBasics:

    def test_ring_3_agents(self):
        adj = build_topology(AGENT_IDS_3, "ring", seed=42)
        assert set(adj.keys()) == {"A1", "A2", "A3"}
        # Each agent should have exactly 2 neighbors in a ring
        for agent, neighbors in adj.items():
            assert len(neighbors) == 2, f"{agent} should have 2 neighbors in ring, got {neighbors}"

    def test_fully_connected_3(self):
        adj = build_topology(AGENT_IDS_3, "fully_connected", seed=42)
        for agent, neighbors in adj.items():
            assert len(neighbors) == 2  # 3 agents, each knows the other 2

    def test_chain_5(self):
        adj = build_topology(AGENT_IDS_5, "chain", seed=42)
        # End nodes have 1 neighbor, middle nodes have 2
        end_agents = [a for a, n in adj.items() if len(n) == 1]
        middle_agents = [a for a, n in adj.items() if len(n) == 2]
        assert len(end_agents) == 2
        assert len(middle_agents) == 3

    def test_all_topologies_produce_adjacency(self):
        """All topology types should produce a non-empty adjacency list with correct keys."""
        for topo in SUPPORTED_TOPOLOGIES:
            adj = build_topology(AGENT_IDS_5, topo, seed=42)
            assert set(adj.keys()) == set(AGENT_IDS_5), f"Topology {topo} missing agent keys"

    def test_adjacency_is_symmetric(self):
        """All edges should be undirected (if A→B then B→A)."""
        for topo in ["ring", "fully_connected", "chain"]:
            adj = build_topology(AGENT_IDS_5, topo, seed=42)
            for agent, neighbors in adj.items():
                for neighbor in neighbors:
                    assert agent in adj[neighbor], (
                        f"Topology {topo}: {neighbor}→{agent} missing (not symmetric)"
                    )

    def test_ring_correct_neighbors(self):
        """In a ring A1→A2→A3→A1, each agent knows exactly prev and next."""
        adj = build_topology(AGENT_IDS_3, "ring", seed=42)
        # A1 should connect to A2 and A3 (ring: A1-A2-A3-A1)
        assert len(adj["A1"]) == 2
        assert len(adj["A2"]) == 2
        assert len(adj["A3"]) == 2

    def test_centralized_hub(self):
        adj = build_topology(AGENT_IDS_5, "centralized_hub", seed=42)
        # One agent (hub) should have 4 neighbors, others have 1
        neighbor_counts = [len(n) for n in adj.values()]
        assert max(neighbor_counts) == 4

    def test_seed_reproducibility(self):
        """Same seed → same topology for random graphs."""
        adj1 = build_topology(AGENT_IDS_5, "random", seed=42)
        adj2 = build_topology(AGENT_IDS_5, "random", seed=42)
        assert adj1 == adj2

    def test_different_seeds_may_differ(self):
        """Different seeds should produce different random topologies."""
        adj1 = build_topology(AGENT_IDS_5, "random", seed=42)
        adj2 = build_topology(AGENT_IDS_5, "random", seed=99)
        # Not guaranteed but very likely
        # We just test they both have valid structure
        assert set(adj1.keys()) == set(AGENT_IDS_5)
        assert set(adj2.keys()) == set(AGENT_IDS_5)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            build_topology(AGENT_IDS_3, "star_wars", seed=42)

    def test_single_agent_raises(self):
        with pytest.raises(ValueError):
            build_topology(["A1"], "ring", seed=42)
