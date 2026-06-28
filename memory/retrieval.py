"""
memory/retrieval.py

Memory retrieval for agents.
Given a query message, finds the top-K most relevant past messages
from an agent's short-term memory.

Relevance is computed via:
  1. TF-IDF keyword overlap (primary, no heavy ML deps)
  2. Claim similarity (exact or substring match bonus)
  3. Evidence overlap (shared document IDs)
  4. Confidence weight (higher confidence → slightly higher score)

Returns top-K messages sorted by descending relevance score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from core.message import Message


def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _tf(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _idf(token: str, corpus: List[List[str]]) -> float:
    """Inverse document frequency of a token across the corpus."""
    n_docs = len(corpus)
    n_containing = sum(1 for doc in corpus if token in doc)
    if n_containing == 0:
        return 0.0
    return math.log((1 + n_docs) / (1 + n_containing)) + 1.0


def tfidf_similarity(query_tokens: List[str], doc_tokens: List[str], corpus: List[List[str]]) -> float:
    """
    Compute TF-IDF cosine similarity between query and a document.
    Simplified: only uses query vocabulary.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    q_tf = _tf(query_tokens)
    d_tf = _tf(doc_tokens)

    dot = 0.0
    q_norm = 0.0
    d_norm = 0.0

    all_tokens = set(query_tokens) | set(doc_tokens)
    for token in all_tokens:
        idf = _idf(token, corpus)
        q_val = q_tf.get(token, 0) * idf
        d_val = d_tf.get(token, 0) * idf
        dot += q_val * d_val
        q_norm += q_val ** 2
        d_norm += d_val ** 2

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def evidence_overlap_score(q_ids: List[str], d_ids: List[str]) -> float:
    """Jaccard similarity between evidence ID sets."""
    q_set = set(q_ids)
    d_set = set(d_ids)
    if not q_set and not d_set:
        return 0.0
    intersection = q_set & d_set
    union = q_set | d_set
    return len(intersection) / len(union)


def claim_similarity_bonus(q_claim: str, d_claim: str) -> float:
    """Bonus score when claims share significant substring overlap."""
    q_words = set(_tokenize(q_claim))
    d_words = set(_tokenize(d_claim))
    if not q_words:
        return 0.0
    overlap = q_words & d_words
    return len(overlap) / len(q_words)


def score_message(
    query: Message,
    candidate: Message,
    corpus: List[List[str]],
    w_tfidf: float = 0.5,
    w_claim: float = 0.3,
    w_evidence: float = 0.15,
    w_confidence: float = 0.05,
) -> float:
    """
    Compute relevance score of a candidate message relative to a query message.

    Weights:
      w_tfidf     — TF-IDF text similarity on claim+reasoning
      w_claim     — exact/substring claim overlap bonus
      w_evidence  — shared evidence document IDs
      w_confidence— confidence weighting (higher = slightly more relevant)
    """
    q_tokens = _tokenize(query.claim + " " + query.reasoning)
    d_tokens = _tokenize(candidate.claim + " " + candidate.reasoning)

    tfidf_score = tfidf_similarity(q_tokens, d_tokens, corpus)
    claim_score = claim_similarity_bonus(query.claim, candidate.claim)
    ev_score = evidence_overlap_score(query.evidence_ids, candidate.evidence_ids)
    conf_score = candidate.confidence

    return (
        w_tfidf * tfidf_score
        + w_claim * claim_score
        + w_evidence * ev_score
        + w_confidence * conf_score
    )


def retrieve_relevant(
    query: Message,
    memory: List[Message],
    top_k: int = 3,
) -> List[Tuple[Message, float]]:
    """
    Retrieve top-K most relevant messages from memory relative to query.

    Args:
        query:  The incoming message to match against
        memory: Agent's past message list
        top_k:  Number of results to return

    Returns:
        List of (message, score) tuples, sorted by descending score.
    """
    if not memory:
        return []

    # Build corpus from all memory for IDF computation
    corpus = [
        _tokenize(m.claim + " " + m.reasoning)
        for m in memory
    ]

    scored = [
        (msg, score_message(query, msg, corpus))
        for msg in memory
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
