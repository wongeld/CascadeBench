"""
core/topology.py

Configurable graph topology engine.
Returns adjacency lists: Dict[str, List[str]]

Supported topologies:
  fully_connected   — every agent connected to every other
  ring              — each agent connected to next and previous
  chain             — linear, each connected to next only
  tree              — balanced binary tree
  random            — Erdos-Renyi random graph
  small_world       — Watts-Strogatz small world
  scale_free        — Barabasi-Albert scale-free
  centralized_hub   — one hub connected to all; others only to hub
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import networkx as nx


TopologyType = str  # one of the topology strings above

SUPPORTED_TOPOLOGIES = [
    "fully_connected",
    "ring",
    "chain",
    "tree",
    "random",
    "small_world",
    "scale_free",
    "centralized_hub",
]


def build_topology(
    agent_ids: List[str],
    topology_type: TopologyType,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, List[str]]:
    """
    Build and return an adjacency list for the given topology.

    Args:
        agent_ids: ordered list of agent ID strings (e.g. ["A1", "A2", "A3"])
        topology_type: one of SUPPORTED_TOPOLOGIES
        seed: for reproducible random graphs
        **kwargs: topology-specific params (e.g. k for small_world, m for scale_free)

    Returns:
        adjacency list: {agent_id: [neighbor_id, ...]}
    """
    n = len(agent_ids)
    if n < 2:
        raise ValueError("Need at least 2 agents to build a topology.")

    if topology_type == "fully_connected":
        G = _fully_connected(n)
    elif topology_type == "ring":
        G = _ring(n)
    elif topology_type == "chain":
        G = _chain(n)
    elif topology_type == "tree":
        G = _tree(n)
    elif topology_type == "random":
        G = _random(n, seed=seed, **kwargs)
    elif topology_type == "small_world":
        G = _small_world(n, seed=seed, **kwargs)
    elif topology_type == "scale_free":
        G = _scale_free(n, seed=seed, **kwargs)
    elif topology_type == "centralized_hub":
        G = _centralized_hub(n)
    else:
        raise ValueError(
            f"Unknown topology: {topology_type!r}. "
            f"Choose from: {SUPPORTED_TOPOLOGIES}"
        )

    # Map integer nodes → agent IDs
    mapping = {i: agent_ids[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Build adjacency list (directed: each agent's neighbors)
    adjacency: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
    for u, v in G.edges():
        if v not in adjacency[u]:
            adjacency[u].append(v)
        if u not in adjacency[v]:
            adjacency[v].append(u)

    return adjacency


# ---------------------------------------------------------------------------
# Topology builders (return undirected networkx graphs with integer nodes)
# ---------------------------------------------------------------------------

def _fully_connected(n: int) -> nx.Graph:
    return nx.complete_graph(n)


def _ring(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def _chain(n: int) -> nx.Graph:
    return nx.path_graph(n)


def _tree(n: int) -> nx.Graph:
    return nx.balanced_tree(r=2, h=_tree_height(n))


def _random(n: int, p: float = 0.4, seed: Optional[int] = None, **kwargs) -> nx.Graph:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure connectivity
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p, seed=(seed or 0) + 1)
    return G


def _small_world(n: int, k: int = 2, p: float = 0.3, seed: Optional[int] = None, **kwargs) -> nx.Graph:
    k = min(k, n - 1)
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def _scale_free(n: int, m: int = 2, seed: Optional[int] = None, **kwargs) -> nx.Graph:
    m = min(m, n - 1)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _centralized_hub(n: int) -> nx.Graph:
    return nx.star_graph(n - 1)  # node 0 is hub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tree_height(n: int) -> int:
    """Find minimum height for balanced binary tree with at least n nodes."""
    import math
    h = 0
    while (2 ** (h + 1) - 1) < n:
        h += 1
    return max(h, 1)


def topology_summary(adjacency: Dict[str, List[str]]) -> str:
    """Human-readable summary of the topology."""
    lines = ["Topology adjacency:"]
    for agent, neighbors in adjacency.items():
        lines.append(f"  {agent} → {neighbors}")
    return "\n".join(lines)
