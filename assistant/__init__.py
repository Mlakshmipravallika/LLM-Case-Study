"""Solari Support Ticket Similarity Assistant — Triage-Retrieve-Verify (TRV) Engine."""

from assistant.preprocessor import extract_facets, normalize_tokens
from assistant.retriever import HybridTicketRetriever
from assistant.triage import SupportAssistantEngine

__all__ = [
    "extract_facets",
    "normalize_tokens",
    "HybridTicketRetriever",
    "SupportAssistantEngine",
]
