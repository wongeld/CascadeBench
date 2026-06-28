"""
experiments/run_experiment.py

CLI entry point for running simulation experiments.

Usage:
  # Dummy mode (no API key, deterministic)
  python experiments/run_experiment.py --config experiments/configs/milestone1.yaml --model dummy

  # OpenRouter
  OPENROUTER_API_KEY=your_key python experiments/run_experiment.py \\
      --config experiments/configs/milestone1.yaml \\
      --model openrouter/meta-llama/llama-3.1-8b-instruct:free

  # Ollama (local)
  python experiments/run_experiment.py \\
      --config experiments/configs/milestone1.yaml \\
      --model ollama/llama3.2

  # NVIDIA API
  NVIDIA_API_KEY=your_key python experiments/run_experiment.py \\
      --config experiments/configs/milestone1.yaml \\
      --model nvidia/mistralai/mistral-medium-3.5-128b

  # Override config params on CLI
  python experiments/run_experiment.py \\
      --config experiments/configs/milestone1.yaml \\
      --model dummy \\
      --seed 123 \\
      --rounds 10 \\
      --topology fully_connected
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from engine.simulator import ExperimentConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a multi-agent LLM simulation experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model string (dummy | openrouter/<model> | ollama/<model> | nvidia/<model>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of rounds.",
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Override topology type.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        dest="experiment_id",
        help="Override experiment ID (affects log directory name).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-event console output.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load base config
    config_dict = load_config(args.config)

    # Apply CLI overrides
    if args.model is not None:
        config_dict["model"] = args.model
    if args.seed is not None:
        config_dict["seed"] = args.seed
    if args.rounds is not None:
        config_dict["num_rounds"] = args.rounds
    if args.topology is not None:
        config_dict["topology"] = args.topology
    if args.experiment_id is not None:
        config_dict["experiment_id"] = args.experiment_id

    # Build config object
    try:
        config = ExperimentConfig.from_dict(config_dict)
    except TypeError as e:
        print(f"[ERROR] Invalid config: {e}")
        sys.exit(1)

    # Run experiment
    result = run_experiment(config)

    print(f"Log directory: {result.log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())