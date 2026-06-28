"""
engine/simulator.py

Main simulation engine. Orchestrates the full experiment pipeline:

  1. Seed RNG
  2. Build environment + apply error injections
  3. Build topology
  4. Instantiate agents + LLM client
  5. Run N rounds of message passing
  6. Compute metrics
  7. Write outputs (logs, belief evolution, metrics, summary)

Entry point: run_experiment(config)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.agent import Agent
from core.environment import Environment, build_milestone1_environment
from core.llm_client import LLMClient
from core.message import Message
from core.topology import build_topology, topology_summary
from engine.message_passing import MessagePassingEngine
from sim_logging.logger import ExperimentLogger
from metrics.cascade_metrics import compute_all_metrics
from utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Experiment Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Full specification for a reproducible experiment.
    Every field is serializable to YAML/JSON.
    """

    experiment_id: str
    seed: int
    num_agents: int
    topology: str                    # one of SUPPORTED_TOPOLOGIES
    num_rounds: int
    model: str                       # "dummy" | "openrouter/<m>" | "ollama/<m>"
    temperature: float = 0.2
    memory_window: int = 10
    retrieval_top_k: int = 3
    belief_update_alpha: float = 0.4
    message_passing_mode: str = "synchronous"
    error_injections: List[dict] = field(default_factory=list)
    target_claim: str = (
        "Vitamin D supplementation reduces the risk of acute respiratory infections."
    )
    # Optional: agent_ids override (default: A1, A2, ...)
    agent_ids: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "topology": self.topology,
            "num_rounds": self.num_rounds,
            "model": self.model,
            "temperature": self.temperature,
            "memory_window": self.memory_window,
            "retrieval_top_k": self.retrieval_top_k,
            "belief_update_alpha": self.belief_update_alpha,
            "message_passing_mode": self.message_passing_mode,
            "error_injections": self.error_injections,
            "target_claim": self.target_claim,
            "agent_ids": self.agent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Experiment Result
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """All outputs from a completed experiment."""

    experiment_id: str
    config: ExperimentConfig
    final_beliefs: Dict[str, Dict[str, float]]   # {agent_id: {claim: conf}}
    belief_evolution: Dict[str, List[Dict]]      # {agent_id: [{round, beliefs}]}
    metrics: dict
    log_dir: str


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Execute a full experiment from config.

    Returns ExperimentResult with all outputs.
    Also writes to logs/<experiment_id>/:
      - trace.jsonl
      - belief_evolution.json
      - metrics.json
      - injection_log.json
      - config.json
      - summary.json
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {config.experiment_id}")
    print(f"  Model: {config.model}  |  Topology: {config.topology}")
    print(f"  Agents: {config.num_agents}  |  Rounds: {config.num_rounds}")
    print(f"  Seed: {config.seed}")
    print(f"{'='*60}\n")

    # 1. Seed everything
    set_global_seed(config.seed)
    rng = random.Random(config.seed)

    # 2. Setup logger
    logger = ExperimentLogger(experiment_id=config.experiment_id, verbose=True)
    logger.log_experiment_start(config.to_dict())

    # 3. Build environment
    env = build_milestone1_environment(rng=rng)

    # 4. Apply error injections (timestep=0 injections happen before round 1)
    t0_injections = [inj for inj in config.error_injections if inj.get("timestep", 0) == 0]
    for inj in t0_injections:
        env.apply_injection(inj, timestep=0)
        for record in env.get_injection_log():
            logger.log_injection(record)

    # Log injection file
    logger.get_store().write_json("injection_log.json", env.get_injection_log())

    # 5. Build topology
    agent_ids = config.agent_ids or [f"A{i+1}" for i in range(config.num_agents)]
    adjacency = build_topology(
        agent_ids=agent_ids,
        topology_type=config.topology,
        seed=config.seed,
    )
    print(topology_summary(adjacency))

    # 6. Instantiate LLM client
    llm_client = LLMClient.from_model_string(config.model, seed=config.seed)

    # 7. Instantiate agents
    agents: Dict[str, Agent] = {}
    for aid in agent_ids:
        agents[aid] = Agent(
            agent_id=aid,
            llm_client=llm_client,
            model=config.model.replace("openrouter/", "").replace("ollama/", ""),
            temperature=config.temperature,
            memory_window=config.memory_window,
            retrieval_top_k=config.retrieval_top_k,
            belief_update_alpha=config.belief_update_alpha,
        )

    # 8. Build message passing engine
    mpe = MessagePassingEngine(
        agents=agents,
        adjacency=adjacency,
        mode=config.message_passing_mode,
        rng=rng,
    )

    # 9. Tracking structures
    belief_evolution: Dict[str, List[dict]] = {aid: [] for aid in agent_ids}

    # 10. Simulation loop
    for round_num in range(1, config.num_rounds + 1):
        logger.log_round_start(round_num)
        print(f"\n--- Round {round_num}/{config.num_rounds} ---")

        # Apply mid-experiment injections for this timestep
        mid_injections = [
            inj for inj in config.error_injections
            if inj.get("timestep", 0) == round_num
        ]
        for inj in mid_injections:
            env.apply_injection(inj, timestep=round_num)
            for record in env.get_injection_log():
                logger.log_injection(record)

        # Each agent runs its reasoning step
        outgoing: Dict[str, Message] = {}
        activation_order = mpe.activation_order()

        for aid in activation_order:
            agent = agents[aid]
            emitted = agent.step(
                round_num=round_num,
                environment=env,
                target_claim=config.target_claim,
                logger=logger,
            )
            if emitted:
                outgoing[aid] = emitted

        # Deliver messages to neighbors
        mpe.deliver(outgoing=outgoing, round_num=round_num, logger=logger)

        # Snapshot belief states for this round
        round_snapshot: Dict[str, Dict[str, float]] = {}
        for aid in agent_ids:
            beliefs = dict(agents[aid].state.beliefs)
            round_snapshot[aid] = beliefs
            belief_evolution[aid].append({
                "round": round_num,
                "beliefs": beliefs,
            })

        logger.log_round_end(round_num, round_snapshot)

    # 11. Collect final state
    final_beliefs: Dict[str, Dict[str, float]] = {
        aid: dict(agents[aid].state.beliefs)
        for aid in agent_ids
    }

    # 12. Compute metrics
    trace = logger.get_store().load_trace()
    metrics = compute_all_metrics(
        trace=trace,
        final_beliefs=final_beliefs,
        ground_truth=env.ground_truth,
        target_claim=config.target_claim,
        injection_log=env.get_injection_log(),
        num_rounds=config.num_rounds,
        agent_ids=agent_ids,
    )

    # 13. Write output files
    store = logger.get_store()
    store.write_json("belief_evolution.json", belief_evolution)
    store.write_json("metrics.json", metrics)
    store.write_json("config.json", config.to_dict())
    store.write_json(
        "final_beliefs.json",
        {aid: {k: round(v, 4) for k, v in b.items()} for aid, b in final_beliefs.items()},
    )

    # 14. Print summary
    summary = _build_summary(
        config=config,
        final_beliefs=final_beliefs,
        metrics=metrics,
        agent_ids=agent_ids,
    )
    store.write_json("summary.json", summary)

    logger.log_experiment_end(summary)
    _print_summary(summary)

    print(f"\nOutputs written to: {store.log_dir}\n")

    return ExperimentResult(
        experiment_id=config.experiment_id,
        config=config,
        final_beliefs=final_beliefs,
        belief_evolution=belief_evolution,
        metrics=metrics,
        log_dir=str(store.log_dir),
    )


# ---------------------------------------------------------------------------
# Summary builder + printer
# ---------------------------------------------------------------------------

def _build_summary(
    config: ExperimentConfig,
    final_beliefs: Dict[str, Dict[str, float]],
    metrics: dict,
    agent_ids: List[str],
) -> dict:
    return {
        "experiment_id": config.experiment_id,
        "config_summary": {
            "topology": config.topology,
            "num_agents": config.num_agents,
            "num_rounds": config.num_rounds,
            "model": config.model,
            "seed": config.seed,
        },
        "final_beliefs": {
            aid: {k: round(v, 4) for k, v in b.items()}
            for aid, b in final_beliefs.items()
        },
        "metrics": metrics,
        "total_rounds": config.num_rounds,
    }


def _print_summary(summary: dict) -> None:
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    print(f"  ID: {summary['experiment_id']}")
    cfg = summary["config_summary"]
    print(f"  Topology: {cfg['topology']} | Agents: {cfg['num_agents']} | Rounds: {cfg['num_rounds']}")
    print(f"  Model: {cfg['model']} | Seed: {cfg['seed']}")
    print("\n  FINAL BELIEFS:")
    for aid, beliefs in summary["final_beliefs"].items():
        print(f"    {aid}:")
        for claim, conf in beliefs.items():
            claim_short = claim[:70] + "..." if len(claim) > 70 else claim
            print(f"      [{conf:.3f}] {claim_short}")
    print("\n  METRICS:")
    for k, v in summary["metrics"].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print("="*60 + "\n")
