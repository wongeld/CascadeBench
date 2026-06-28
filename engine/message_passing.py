"""
engine/message_passing.py

Routes messages between agents based on the topology adjacency list.

Modes:
  synchronous  — all agents process simultaneously; state updates at round end
  asynchronous — agents activate in random order; immediate state update

Milestone 1 uses synchronous mode.
"""

from __future__ import annotations

import random
from typing import Dict, List

from core.agent import Agent
from core.message import Message
from sim_logging.logger import ExperimentLogger


class MessagePassingEngine:
    """
    Handles message routing for one simulation round.

    Responsibilities:
      - Deliver outgoing messages from each agent to its neighbors
      - Respect topology adjacency constraints
      - Support synchronous and asynchronous activation modes
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        adjacency: Dict[str, List[str]],
        mode: str = "synchronous",
        rng: random.Random = None,
    ):
        self.agents = agents
        self.adjacency = adjacency
        self.mode = mode
        self._rng = rng or random.Random(0)

    def deliver(
        self,
        outgoing: Dict[str, Message],
        round_num: int,
        logger: ExperimentLogger,
    ) -> None:
        """
        Deliver messages from each agent to its neighbors.

        Args:
            outgoing:  {agent_id: Message} — each agent's emitted message
            round_num: current simulation round
            logger:    for logging send/receive events
        """
        for source_id, message in outgoing.items():
            neighbors = self.adjacency.get(source_id, [])
            for target_id in neighbors:
                if target_id not in self.agents:
                    continue
                # Log the send event
                logger.log_message_sent(
                    round_num=round_num,
                    source_agent=source_id,
                    target_agent=target_id,
                    message_dict=message.to_dict(),
                )
                # Deliver to target inbox
                self.agents[target_id].receive_messages([message])

    def activation_order(self) -> List[str]:
        """
        Return agent activation order for the current round.

        synchronous:  fixed alphabetical order (deterministic)
        asynchronous: shuffled order (seeded RNG)
        """
        agent_ids = sorted(self.agents.keys())
        if self.mode == "asynchronous":
            self._rng.shuffle(agent_ids)
        return agent_ids

    def __repr__(self) -> str:
        return (
            f"MessagePassingEngine(mode={self.mode}, "
            f"agents={list(self.agents.keys())}, "
            f"topology={dict(self.adjacency)})"
        )
