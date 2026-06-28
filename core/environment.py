"""
core/environment.py

Environment module: document store, ground truth, and error injection.

Documents have the form:
{
  "doc_id": "D1",
  "content": "...",
  "source_type": "paper | tweet | blog | report",
  "reliability": 0.0–1.0
}

ErrorInjector applies controlled perturbations and logs each injection.
Every injection is traceable: type, timestep, affected doc, original vs corrupted.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Document schema
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    content: str
    source_type: str  # paper | tweet | blog | report
    reliability: float  # 0.0–1.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source_type": self.source_type,
            "reliability": self.reliability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(**data)


# ---------------------------------------------------------------------------
# Injection event log entry
# ---------------------------------------------------------------------------

@dataclass
class InjectionRecord:
    injection_type: str
    doc_id: str
    timestep: int
    original: str
    corrupted: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "injection_type": self.injection_type,
            "doc_id": self.doc_id,
            "timestep": self.timestep,
            "original": self.original,
            "corrupted": self.corrupted,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Error Injector
# ---------------------------------------------------------------------------

class ErrorInjector:
    """
    Applies controlled perturbations to documents.

    Supported injection types:
      flip_sentence          — negates or contradicts a sentence
      biased_claim           — shifts claim toward a specific bias
      missing_evidence       — removes key supporting sentence
      fabricated_social_post — appends a fake authoritative tweet
      retracted_paper_signal — adds retraction notice
      confidence_distortion  — inflates/deflates reliability score
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random(0)
        self.injection_log: List[InjectionRecord] = []

    def apply(
        self,
        documents: List[Document],
        injection_config: dict,
        timestep: int = 0,
    ) -> List[Document]:
        """
        Apply a single injection specified by injection_config dict.

        injection_config keys:
          type         — injection type string
          doc_id       — target document ID
          (type-specific params)
        """
        docs = {d.doc_id: d for d in documents}
        itype = injection_config["type"]
        doc_id = injection_config["doc_id"]

        if doc_id not in docs:
            raise ValueError(f"Document {doc_id!r} not found in environment.")

        doc = docs[doc_id]
        original_content = doc.content

        if itype == "flip_sentence":
            doc = self._flip_sentence(doc, injection_config)
        elif itype == "biased_claim":
            doc = self._biased_claim(doc, injection_config)
        elif itype == "missing_evidence":
            doc = self._missing_evidence(doc, injection_config)
        elif itype == "fabricated_social_post":
            doc = self._fabricated_social_post(doc, injection_config)
        elif itype == "retracted_paper_signal":
            doc = self._retracted_paper_signal(doc, injection_config)
        elif itype == "confidence_distortion":
            doc = self._confidence_distortion(doc, injection_config)
        else:
            raise ValueError(f"Unknown injection type: {itype!r}")

        record = InjectionRecord(
            injection_type=itype,
            doc_id=doc_id,
            timestep=timestep,
            original=original_content,
            corrupted=doc.content,
            metadata={k: v for k, v in injection_config.items() if k not in ("type", "doc_id")},
        )
        self.injection_log.append(record)

        # Reconstruct list with patched doc
        return [docs[d.doc_id] if d.doc_id != doc_id else doc for d in documents]

    # ---- Injection implementations ----------------------------------------

    def _flip_sentence(self, doc: Document, cfg: dict) -> Document:
        """Negate the sentence at target_sentence_idx."""
        sentences = [s.strip() for s in doc.content.split(".") if s.strip()]
        idx = cfg.get("target_sentence_idx", 0)
        if idx >= len(sentences):
            idx = 0

        original_sentence = sentences[idx]
        # Simple negation heuristics
        if " does not " in original_sentence or " cannot " in original_sentence:
            flipped = original_sentence.replace(" does not ", " does ").replace(" cannot ", " can ")
        elif " not " in original_sentence:
            flipped = original_sentence.replace(" not ", " ")
        elif " is " in original_sentence:
            flipped = original_sentence.replace(" is ", " is not ", 1)
        elif " are " in original_sentence:
            flipped = original_sentence.replace(" are ", " are not ", 1)
        elif " increases " in original_sentence:
            flipped = original_sentence.replace(" increases ", " decreases ", 1)
        elif " decreases " in original_sentence:
            flipped = original_sentence.replace(" decreases ", " increases ", 1)
        elif " causes " in original_sentence:
            flipped = original_sentence.replace(" causes ", " does not cause ", 1)
        else:
            flipped = "It is false that " + original_sentence[0].lower() + original_sentence[1:]

        sentences[idx] = flipped
        doc = copy.copy(doc)
        doc.content = ". ".join(sentences) + "."
        return doc

    def _biased_claim(self, doc: Document, cfg: dict) -> Document:
        """Append a biased framing sentence."""
        bias_text = cfg.get(
            "bias_text",
            " However, recent controversial studies suggest the opposite may be true."
        )
        doc = copy.copy(doc)
        doc.content = doc.content.rstrip(".") + bias_text
        return doc

    def _missing_evidence(self, doc: Document, cfg: dict) -> Document:
        """Remove the sentence at removal_sentence_idx."""
        sentences = [s.strip() for s in doc.content.split(".") if s.strip()]
        idx = cfg.get("removal_sentence_idx", -1)
        if idx < 0:
            idx = len(sentences) - 1
        if sentences:
            sentences.pop(min(idx, len(sentences) - 1))
        doc = copy.copy(doc)
        doc.content = ". ".join(sentences) + ("." if sentences else "")
        return doc

    def _fabricated_social_post(self, doc: Document, cfg: dict) -> Document:
        """Append a fabricated authoritative social media reference."""
        handle = cfg.get("handle", "@ExpertResearcher")
        claim = cfg.get("claim_text", "New findings completely overturn previous consensus.")
        doc = copy.copy(doc)
        doc.content += (
            f" [Social media, {handle}]: \"{claim}\" "
            f"(Source: unverified tweet, reliability: LOW)"
        )
        doc.source_type = "tweet"
        doc.reliability = min(doc.reliability, 0.2)
        return doc

    def _retracted_paper_signal(self, doc: Document, cfg: dict) -> Document:
        """Prepend a retraction notice to the document."""
        notice = cfg.get(
            "notice",
            "[RETRACTION NOTICE] This paper has been retracted due to data irregularities. "
            "Results should not be cited."
        )
        doc = copy.copy(doc)
        doc.content = notice + " " + doc.content
        doc.reliability = 0.0
        return doc

    def _confidence_distortion(self, doc: Document, cfg: dict) -> Document:
        """Distort the reliability score of a document."""
        new_reliability = cfg.get("new_reliability", 0.95)
        doc = copy.copy(doc)
        doc.reliability = float(new_reliability)
        return doc


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Holds the document store, ground truth answers, and injection log.
    Provides document access for agents during reasoning steps.
    """

    def __init__(
        self,
        documents: List[Document],
        ground_truth: Dict[str, str],
        rng: Optional[random.Random] = None,
    ):
        self.documents: List[Document] = documents
        self.ground_truth: Dict[str, str] = ground_truth
        self._rng = rng or random.Random(0)
        self.injector = ErrorInjector(rng=self._rng)

    @property
    def doc_index(self) -> Dict[str, Document]:
        return {d.doc_id: d for d in self.documents}

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.doc_index.get(doc_id)

    def get_documents_for_agent(self, agent_id: str, max_docs: int = 3) -> List[Document]:
        """
        Return a subset of documents for an agent to read.
        Simple strategy: return up to max_docs highest-reliability documents.
        """
        sorted_docs = sorted(self.documents, key=lambda d: d.reliability, reverse=True)
        return sorted_docs[:max_docs]

    def apply_injection(self, injection_config: dict, timestep: int = 0) -> None:
        """Apply an error injection to the document store (mutates documents)."""
        self.documents = self.injector.apply(self.documents, injection_config, timestep)

    def apply_all_injections(self, injections: List[dict]) -> None:
        """Apply a list of injection configs (at their specified timesteps)."""
        for inj in injections:
            ts = inj.get("timestep", 0)
            self.apply_injection(inj, timestep=ts)

    def get_injection_log(self) -> List[dict]:
        return [r.to_dict() for r in self.injector.injection_log]

    def to_dict(self) -> dict:
        return {
            "documents": [d.to_dict() for d in self.documents],
            "ground_truth": self.ground_truth,
        }


# ---------------------------------------------------------------------------
# Built-in Milestone 1 document set
# ---------------------------------------------------------------------------

def build_milestone1_environment(rng: Optional[random.Random] = None) -> Environment:
    """
    Builds the Milestone 1 document set.
    Topic: Vitamin D and immune function (concrete scientific domain).
    Ground truth: vitamin D supplementation does reduce respiratory infection risk.
    D2 will be the injection target (flip_sentence on sentence 0).
    """
    documents = [
        Document(
            doc_id="D1",
            content=(
                "Vitamin D plays a critical role in regulating the immune system. "
                "It activates T-cells and B-cells, which are essential for fighting infections. "
                "Deficiency in vitamin D is associated with increased susceptibility to respiratory infections."
            ),
            source_type="paper",
            reliability=0.95,
        ),
        Document(
            doc_id="D2",
            content=(
                "Vitamin D supplementation significantly reduces the risk of acute respiratory infections. "
                "A meta-analysis of 25 randomized controlled trials found a 12% reduction in infection risk. "
                "The protective effect is strongest in individuals who are severely deficient."
            ),
            source_type="paper",
            reliability=0.90,
        ),
        Document(
            doc_id="D3",
            content=(
                "The immune-modulatory effects of vitamin D include enhancement of innate immunity. "
                "Calcitriol, the active form of vitamin D, induces production of antimicrobial peptides. "
                "These peptides are the first line of defense against pathogens."
            ),
            source_type="report",
            reliability=0.88,
        ),
        Document(
            doc_id="D4",
            content=(
                "Some observational studies show correlation between low vitamin D levels and COVID-19 severity. "
                "However, correlation does not imply causation and confounding factors may exist. "
                "Randomized controlled trials are needed to establish causal relationships."
            ),
            source_type="blog",
            reliability=0.60,
        ),
        Document(
            doc_id="D5",
            content=(
                "Daily vitamin D intake recommendations vary by age and health status. "
                "Most health authorities recommend 600-800 IU per day for adults. "
                "Toxicity can occur at very high doses above 10,000 IU per day."
            ),
            source_type="report",
            reliability=0.85,
        ),
    ]

    ground_truth = {
        "Does vitamin D supplementation reduce respiratory infection risk?": "yes",
        "primary_claim": "Vitamin D supplementation reduces the risk of acute respiratory infections.",
    }

    return Environment(documents=documents, ground_truth=ground_truth, rng=rng)
