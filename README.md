# CascadeBench

**CascadeBench** is an experimental benchmark and simulation framework for studying **error propagation**, **false consensus**, and **epistemic collapse** in decentralized Large Language Model (LLM) multi-agent systems.

Rather than focusing on improving agent performance, CascadeBench is designed as a research platform for understanding how information propagates through collaborative reasoning networks and how small reasoning or factual errors evolve into system-wide failures.

The framework enables controlled experiments by simulating multiple communicating LLM agents, configurable communication topologies, structured memory, and reproducible error injection.

---

## Research Motivation

Large Language Model Multi-Agent Systems (LLM-MAS) have demonstrated promising performance on collaborative reasoning tasks. However, little is known about how incorrect information propagates through decentralized agent networks.

CascadeBench aims to answer questions such as:

* How does a single incorrect claim spread through a swarm?
* Under what conditions does a false consensus emerge?
* Which communication topologies are most robust?
* How does memory influence misinformation persistence?
* Can early indicators predict an impending cascade?

The framework is intended for reproducible scientific experiments rather than production agent deployment.

---

## Features

* Modular LLM agent simulator
* Configurable communication topologies
* Structured inter-agent communication
* Agent memory layer
* Belief graph representation
* Controlled error injection
* Experiment replay
* Full event logging
* Propagation analysis
* Benchmark metrics

---

## Planned Error Injection Strategies

* Sentence modification
* Biased evidence
* Fabricated social media post
* Missing evidence
* Confidence manipulation
* Retracted publication simulation
* Outdated information

---

## Core Components

```text
Environment
        │
        ▼
Document Distribution
        │
        ▼
Agent Network
        │
        ▼
Structured Communication
        │
        ▼
Memory Layer
        │
        ▼
Belief Update
        │
        ▼
Logging
        │
        ▼
Metrics & Analysis
```

---

## Repository Structure

```text
cascadebench/
│
├── core/
├── engine/
├── memory/
├── logging/
├── metrics/
├── experiments/
├── datasets/
├── configs/
├── docs/
└── tests/
```

---

## Current Status

This project is under active research and development.

The initial milestone is to implement:

* Three communicating agents
* Ring topology
* Structured messaging
* Belief tracking
* Error injection
* Complete experiment logging

before expanding to larger-scale experiments.

---

## Long-Term Vision

CascadeBench is intended to become a reusable research benchmark for evaluating robustness and information propagation in collaborative LLM systems across multiple domains including:

* Scientific literature review
* Multi-document reasoning
* News synthesis
* Social media analysis
* Evidence integration

---

## License

MIT License

---

## Citation

Citation information will be added after publication.
