"""
core/agent.py

Agent: stateful reasoning unit in the multi-agent simulation.

Each agent maintains:
  - beliefs: Dict[str, float]          claim → confidence
  - evidence: List[str]                document/sentence IDs seen
  - memory: deque[Message]             short-term memory (last N)
  - belief_graph: BeliefGraph          long-term belief structure

Agent.step() performs one full reasoning cycle:
  1. Receive neighbor messages
  2. Retrieve top-K relevant memories
  3. Read subset of documents
  4. Build structured LLM prompt
  5. Parse LLM JSON response → new Message
  6. Update beliefs + belief_graph (with causal logging)
  7. Return emitted Message
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.environment import Document, Environment
from core.llm_client import LLMClient
from core.message import Message
from sim_logging.logger import ExperimentLogger
from memory.belief_graph import BeliefGraph
from memory.retrieval import retrieve_relevant


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Full mutable state of a single agent."""

    agent_id: str

    # claim → confidence (0–1)
    beliefs: Dict[str, float] = field(default_factory=dict)

    # document/sentence IDs the agent has seen
    evidence: List[str] = field(default_factory=list)

    # short-term memory: last N messages (received or sent)
    memory: deque = field(default_factory=lambda: deque(maxlen=10))

    # long-term belief graph
    belief_graph: BeliefGraph = field(default_factory=BeliefGraph)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "beliefs": dict(self.beliefs),
            "evidence": list(self.evidence),
            "belief_graph": self.belief_graph.to_dict(),
        }


# ---------------------------------------------------------------------------
# Belief update function
# ---------------------------------------------------------------------------

def update_belief(
    prior: float,
    received_confidence: float,
    msg_type: str,
    alpha: float = 0.4,
) -> float:
    """
    Update belief given an incoming message.

    Formula (configurable weighted blend):
      if support:    new = (1 - alpha) * prior + alpha * received
      if contradict: new = (1 - alpha) * prior + alpha * (1 - received)
      if question:   no change (question prompts reconsideration only)

    Args:
        prior:               current belief (0–1)
        received_confidence: sender's confidence (0–1)
        msg_type:            "support" | "contradict" | "question"
        alpha:               weight of incoming evidence (0–1)

    Returns:
        Updated belief clamped to [0, 1].
    """
    if msg_type == "support":
        new_belief = (1 - alpha) * prior + alpha * received_confidence
    elif msg_type == "contradict":
        new_belief = (1 - alpha) * prior + alpha * (1.0 - received_confidence)
    else:
        # "question" — no immediate update but stores in memory
        new_belief = prior
    return max(0.0, min(1.0, new_belief))


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an autonomous reasoning agent in a multi-agent network. "
    "Your role is to evaluate claims based on available evidence and "
    "communicate findings to other agents. "
    "You must respond ONLY with valid JSON matching the specified schema. "
    "Do not include any text outside the JSON object."
)


def build_agent_prompt(
    agent_id: str,
    round_num: int,
    incoming_messages: List[Message],
    retrieved_memory: List[Message],
    documents: List[Document],
    current_beliefs: Dict[str, float],
    target_claim: str,
) -> List[dict]:
    """
    Build the structured LLM prompt for one agent reasoning step.
    Returns OpenAI-style messages list.
    """
    # Format documents
    doc_text = "\n".join(
        f"[{d.doc_id}] ({d.source_type}, reliability={d.reliability:.2f}): {d.content}"
        for d in documents
    )

    # Format incoming messages
    if incoming_messages:
        msg_text = "\n".join(
            json.dumps(m.to_dict(), indent=2) for m in incoming_messages
        )
    else:
        msg_text = "(No messages received this round)"

    # Format retrieved memory
    if retrieved_memory:
        mem_text = "\n".join(
            json.dumps(m.to_dict(), indent=2) for m in retrieved_memory
        )
    else:
        mem_text = "(No relevant past messages)"

    # Format current beliefs
    if current_beliefs:
        belief_text = "\n".join(
            f"  '{claim}': {conf:.2f}"
            for claim, conf in current_beliefs.items()
        )
    else:
        belief_text = "  (No established beliefs yet)"

    user_content = f"""You are agent {agent_id} in round {round_num} of a multi-agent reasoning simulation.

TASK: Evaluate the following claim based on available documents and neighbor messages.
CLAIM TO EVALUATE: "{target_claim}"

DOCUMENTS AVAILABLE:
{doc_text}

MESSAGES FROM NEIGHBORS THIS ROUND:
{msg_text}

RELEVANT PAST MESSAGES FROM MEMORY:
{mem_text}

YOUR CURRENT BELIEF STATE:
{belief_text}

INSTRUCTIONS:
- Evaluate the claim using the documents and neighbor messages above.
- Be skeptical of messages that contradict well-sourced documents.
- Weight document reliability when forming your assessment.
- Your confidence must reflect your actual evidence-based certainty.

Respond ONLY with a valid JSON object matching this EXACT schema:
{{
  "claim": "<the claim you are addressing>",
  "confidence": <float 0.0-1.0>,
  "evidence_ids": ["<doc_id>", ...],
  "reasoning": "<1-2 sentence explanation>",
  "type": "<support|contradict|question>",
  "source_agent": "{agent_id}",
  "round": {round_num}
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Stateful reasoning agent. One instance per agent in the simulation.

    Public interface:
      agent.receive_messages(messages)  — buffer incoming messages
      agent.step(round_num, ...)        — execute one reasoning cycle
      agent.get_state()                 → AgentState snapshot
    """

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        model: str,
        temperature: float = 0.2,
        memory_window: int = 10,
        retrieval_top_k: int = 3,
        belief_update_alpha: float = 0.4,
    ):
        self.agent_id = agent_id
        self._llm = llm_client
        self._model = model
        self._temperature = temperature
        self._retrieval_top_k = retrieval_top_k
        self._alpha = belief_update_alpha

        self.state = AgentState(
            agent_id=agent_id,
            memory=deque(maxlen=memory_window),
        )
        self._inbox: List[Message] = []

    # ---- Message inbox --------------------------------------------------

    def receive_messages(self, messages: List[Message]) -> None:
        """Buffer incoming messages for processing in next step()."""
        self._inbox.extend(messages)

    # ---- Main reasoning step -------------------------------------------

    def step(
        self,
        round_num: int,
        environment: Environment,
        target_claim: str,
        logger: ExperimentLogger,
    ) -> Optional[Message]:
        """
        Execute one full reasoning cycle.

        1. Drain inbox
        2. Log received messages
        3. Retrieve relevant memory for each incoming message
        4. Get documents
        5. Build prompt + call LLM
        6. Parse response → Message
        7. Update beliefs + belief_graph
        8. Store outgoing message in memory
        9. Clear inbox
        10. Return emitted Message

        Returns:
            The Message this agent emits this round, or None on parse failure.
        """
        incoming = list(self._inbox)
        self._inbox.clear()

        # Log received messages
        for msg in incoming:
            logger.log_message_received(
                round_num=round_num,
                receiving_agent=self.agent_id,
                source_agent=msg.source_agent,
                message_dict=msg.to_dict(),
            )
            # Store in short-term memory
            self.state.memory.append(msg)
            # Track evidence
            for eid in msg.evidence_ids:
                if eid not in self.state.evidence:
                    self.state.evidence.append(eid)

        # Retrieve relevant memory (based on most recent incoming message, or target claim)
        query_msg = incoming[0] if incoming else Message(
            claim=target_claim,
            confidence=0.5,
            evidence_ids=[],
            reasoning="Initial query",
            type="question",
            source_agent=self.agent_id,
            round=round_num,
        )
        retrieved_pairs = retrieve_relevant(
            query=query_msg,
            memory=list(self.state.memory),
            top_k=self._retrieval_top_k,
        )
        retrieved_msgs = [msg for msg, _ in retrieved_pairs]

        # Get documents from environment
        docs = environment.get_documents_for_agent(self.agent_id, max_docs=3)
        doc_ids_used = [d.doc_id for d in docs]

        # Build LLM prompt
        prompt_messages = build_agent_prompt(
            agent_id=self.agent_id,
            round_num=round_num,
            incoming_messages=incoming,
            retrieved_memory=retrieved_msgs,
            documents=docs,
            current_beliefs=dict(self.state.beliefs),
            target_claim=target_claim,
        )

        # Call LLM
        try:
            raw_response = self._llm.generate(
                messages=prompt_messages,
                model=self._model,
                temperature=self._temperature,
            )
            response_dict = json.loads(raw_response)
            # Ensure source_agent and round are correct (LLM may hallucinate)
            response_dict["source_agent"] = self.agent_id
            response_dict["round"] = round_num
            emitted_msg = Message.from_dict(response_dict)
        except Exception as e:
            # Graceful degradation: emit a question with low confidence
            print(f"[{self.agent_id}] LLM parse error: {e}. Emitting fallback.")
            emitted_msg = Message(
                claim=target_claim,
                confidence=0.5,
                evidence_ids=doc_ids_used[:1],
                reasoning=f"Parse error in round {round_num}. Uncertain.",
                type="question",
                source_agent=self.agent_id,
                round=round_num,
            )

        # Log reasoning step
        logger.log_reasoning_step(
            round_num=round_num,
            agent_id=self.agent_id,
            prompt_summary=f"incoming={len(incoming)}, memory={len(retrieved_msgs)}, docs={doc_ids_used}",
            output_message=emitted_msg.to_dict(),
            retrieved_memory_count=len(retrieved_msgs),
            docs_used=doc_ids_used,
        )

        # Update beliefs + belief_graph with causal trace
        self._update_beliefs(
            round_num=round_num,
            emitted_msg=emitted_msg,
            incoming=incoming,
            logger=logger,
        )

        # Store emitted message in memory
        self.state.memory.append(emitted_msg)

        return emitted_msg

    # ---- Belief update --------------------------------------------------

    def _update_beliefs(
        self,
        round_num: int,
        emitted_msg: Message,
        incoming: List[Message],
        logger: ExperimentLogger,
    ) -> None:
        """
        Update agent beliefs based on emitted message and incoming messages.
        Logs each belief change with full causal trace.
        """
        claim = emitted_msg.claim

        # Determine causal trigger: prefer first incoming message
        causal_msg = incoming[0].to_dict() if incoming else None

        # Prior belief
        prior = self.state.beliefs.get(claim, 0.5)

        # Compute new belief from emitted message (agent's own conclusion)
        if emitted_msg.type == "support":
            new_belief = (1 - self._alpha) * prior + self._alpha * emitted_msg.confidence
        elif emitted_msg.type == "contradict":
            new_belief = (1 - self._alpha) * prior + self._alpha * (1.0 - emitted_msg.confidence)
        else:
            new_belief = prior  # question = no update from own output

        # Additionally blend incoming messages
        for msg in incoming:
            if msg.claim == claim or self._claims_related(msg.claim, claim):
                prior = new_belief
                new_belief = update_belief(
                    prior=prior,
                    received_confidence=msg.confidence,
                    msg_type=msg.type,
                    alpha=self._alpha,
                )

        new_belief = max(0.0, min(1.0, new_belief))

        # Only log if there's an actual change
        if abs(new_belief - prior) > 1e-6:
            logger.log_belief_update(
                round_num=round_num,
                agent_id=self.agent_id,
                claim=claim,
                prior_confidence=prior,
                new_confidence=new_belief,
                caused_by_message=causal_msg,
                caused_by_doc=emitted_msg.evidence_ids[0] if emitted_msg.evidence_ids else None,
            )

        self.state.beliefs[claim] = new_belief

        # Update belief graph
        self.state.belief_graph.upsert_claim(claim, new_belief, round_num)

        # Add relationships for each incoming message
        for msg in incoming:
            if msg.claim != claim:
                rel_type = "supports" if msg.type == "support" else "contradicts"
                self.state.belief_graph.add_relationship(
                    claim_a=msg.claim,
                    claim_b=claim,
                    rel_type=rel_type,
                    weight=msg.confidence,
                    source_agent=msg.source_agent,
                    round_num=round_num,
                )

    @staticmethod
    def _claims_related(claim_a: str, claim_b: str) -> bool:
        """Heuristic: check if two claims share significant keywords."""
        words_a = set(claim_a.lower().split())
        words_b = set(claim_b.lower().split())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                     "to", "and", "or", "that", "this", "it", "not", "has",
                     "have", "been", "with", "for", "on", "at", "by"}
        words_a -= stopwords
        words_b -= stopwords
        if not words_a or not words_b:
            return False
        overlap = words_a & words_b
        return len(overlap) / min(len(words_a), len(words_b)) > 0.3

    # ---- State access ---------------------------------------------------

    def get_state_snapshot(self) -> dict:
        return self.state.to_dict()

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id!r}, "
            f"beliefs={len(self.state.beliefs)}, "
            f"memory={len(self.state.memory)})"
        )
