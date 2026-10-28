"""
Upgraded utility module (backward-compatible with the starter kit signature).

Fixes the starter bugs:
1. Contraction handling ("can't" -> "cannot" instead of ["can", "t"])
2. Stopword removal (filters conversational noise like "customer", "the", "a")
3. Stemming & tag/category inclusion in similarity scoring
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

# Ensure root package is importable whether run from project root or starter_kit/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from assistant.preprocessor import normalize_tokens
from assistant.retriever import HybridTicketRetriever


def load_tickets(path: str) -> List[Dict[str, Any]]:
    """Load a list of ticket dicts from a JSON file shaped like data/tickets.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tickets", data.get("new_tickets", []))


def normalize_text(text: str) -> Set[str]:
    """Lowercase, expand contractions, remove stopwords, stem, and return token set."""
    return set(normalize_tokens(text, include_concepts=True))


def keyword_overlap_score(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Cosine-normalized overlap similarity between two token sets, in [0, 1]."""
    if not tokens_a or not tokens_b:
        return 0.0
    shared = tokens_a & tokens_b
    return len(shared) / ((len(tokens_a) ** 0.5) * (len(tokens_b) ** 0.5))


def find_top_matches(
    new_ticket_text: str,
    historical_tickets: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[Tuple[Dict[str, Any], float]]:
    """Rank historical tickets using the HybridTicketRetriever while preserving starter return signature."""
    retriever = HybridTicketRetriever(historical_tickets, resolved_only=True)
    result = retriever.retrieve(subject=new_ticket_text, description="", top_n=top_n)
    scored: List[Tuple[Dict[str, Any], float]] = []
    for cand in result["candidates"]:
        orig_ticket = retriever.all_tickets_by_id[cand["ticket_id"]]
        scored.append((orig_ticket, cand["score"]))
    return scored
