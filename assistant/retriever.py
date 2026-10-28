"""
Stage 2: Multi-Lane Hybrid Ticket Retriever with Diagnostic Facet Alignment & Root-Cause Graph.

Combines three complementary retrieval lanes via Reciprocal Rank Fusion (RRF) + Calibrated Scoring:
1. Lane 1 (Lexical): Field-weighted BM25 over Subject (3.0x), Tags/Category (2.5x),
   Description (1.5x), and Resolution Notes (1.0x).
2. Lane 2 (Semantic / Subword): Domain Concept Cluster alignment + Character 4-gram cosine similarity.
3. Lane 3 (Graph & Facets): Orthogonal WHERE (entity) x WHAT (symptom) facet verification and
   cross-ticket root-cause graph links (e.g., T-1401 <-> T-1104).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from assistant.preprocessor import (
    char_ngrams,
    extract_facets,
    normalize_tokens,
    preprocess_raw_text,
    stem_token,
    STOPWORDS,
)

CATEGORY_LABELS: Dict[str, str] = {
    "auth": "Authentication & Login",
    "sync": "Data Sync & Integrations",
    "render": "Dashboard & Chart Rendering",
    "billing": "Billing, Invoices & Payments",
    "export": "Reports & Data Exports",
    "perm": "Roles & Permissions",
    "perf": "Performance & Latency",
    "api": "API & Webhooks",
}


class HybridTicketRetriever:
    """Indexes historical resolved support tickets across lexical, semantic, facet, and graph lanes."""

    FIELD_WEIGHTS = {
        "subject": 3.0,
        "tags": 2.5,
        "description": 1.6,
        "resolution_notes": 1.0,
    }

    def __init__(self, tickets: List[Dict[str, Any]], resolved_only: bool = True) -> None:
        self.all_tickets_by_id: Dict[str, Dict[str, Any]] = {
            t["ticket_id"]: t for t in tickets if "ticket_id" in t
        }
        if resolved_only:
            self.tickets: List[Dict[str, Any]] = [
                t for t in tickets if t.get("status", "resolved").lower() == "resolved"
            ]
        else:
            self.tickets = list(tickets)

        self.doc_tokens: List[Counter[str]] = []
        self.doc_lengths: List[float] = []
        self.doc_ngrams: List[Set[str]] = []
        self.doc_facets: List[Dict[str, List[str]]] = []
        self.df: Counter[str] = Counter()
        self.corpus_vocab: Set[str] = set()
        self.category_token_weights: Dict[str, Counter[str]] = defaultdict(Counter)
        self.root_cause_graph: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        self._build_index()

    def _build_index(self) -> None:
        """Build weighted BM25 index, corpus vocabulary, category profiles, and root-cause graph."""
        self.doc_tokens.clear()
        self.doc_lengths.clear()
        self.doc_ngrams.clear()
        self.doc_facets.clear()
        self.df.clear()
        self.corpus_vocab.clear()
        self.category_token_weights.clear()
        self.root_cause_graph.clear()

        # First pass: collect raw + stemmed vocabulary across all historical tickets
        for t in self.tickets:
            raw_text = " ".join([
                t.get("subject", ""),
                t.get("description", ""),
                t.get("resolution_notes", ""),
                " ".join(t.get("tags", [])),
                t.get("category", ""),
            ])
            cleaned = preprocess_raw_text(raw_text)
            for w in re.sub(r"[^a-z0-9\s]", " ", cleaned).split():
                if w not in STOPWORDS and len(w) > 1:
                    self.corpus_vocab.add(w)
                    self.corpus_vocab.add(stem_token(w))

        # Second pass: build per-ticket weighted term frequencies and facet signatures
        for t in self.tickets:
            weighted_counts: Counter[str] = Counter()

            subj_tokens = normalize_tokens(t.get("subject", ""), include_concepts=True)
            for tok in subj_tokens:
                weighted_counts[tok] += self.FIELD_WEIGHTS["subject"]

            tag_text = " ".join(t.get("tags", [])) + " " + t.get("category", "")
            tag_tokens = normalize_tokens(tag_text, include_concepts=True)
            for tok in tag_tokens:
                weighted_counts[tok] += self.FIELD_WEIGHTS["tags"]

            desc_tokens = normalize_tokens(t.get("description", ""), include_concepts=True)
            for tok in desc_tokens:
                weighted_counts[tok] += self.FIELD_WEIGHTS["description"]

            res_tokens = normalize_tokens(t.get("resolution_notes", ""), include_concepts=True)
            for tok in res_tokens:
                weighted_counts[tok] += self.FIELD_WEIGHTS["resolution_notes"]

            self.doc_tokens.append(weighted_counts)
            self.doc_lengths.append(sum(weighted_counts.values()) or 1.0)

            for tok in weighted_counts:
                self.df[tok] += 1

            cat = t.get("category", "unknown")
            for tok, weight in weighted_counts.items():
                self.category_token_weights[cat][tok] += weight

            ngram_source = f"{t.get('subject', '')} {t.get('description', '')} {' '.join(t.get('tags', []))}"
            self.doc_ngrams.append(char_ngrams(ngram_source, n=4))
            self.doc_facets.append(
                extract_facets(t.get("subject", ""), t.get("description", ""), self.corpus_vocab)
            )

        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        )

        # Build Cross-Ticket Root-Cause Graph (explicit ID mentions + shared systemic root-cause tags)
        self._build_root_cause_graph()

    def _build_root_cause_graph(self) -> None:
        """Link tickets that explicitly reference each other (e.g., T-1401 -> T-1104) or share systemic causes."""
        # 1. Explicit ticket ID mentions in resolution_notes or description
        for t in self.tickets:
            src_id = t["ticket_id"]
            notes = f"{t.get('description', '')} {t.get('resolution_notes', '')}"
            for match in re.finditer(r"\b(T-\d{4})\b", notes):
                target_id = match.group(1)
                if target_id != src_id and target_id in self.all_tickets_by_id:
                    target = self.all_tickets_by_id[target_id]
                    self._add_graph_edge(
                        src_id,
                        target_id,
                        f"Explicitly cross-referenced in resolution notes ({src_id} ↔ {target_id}: shared UTC/timezone root cause)",
                    )
                    self._add_graph_edge(
                        target_id,
                        src_id,
                        f"Referenced by {src_id} ({target.get('category', '')} → {t.get('category', '')} shared UTC/timezone fix)",
                    )

        # 2. Shared high-signal systemic tags across different categories (e.g., timezone, caching, sso, scheduled-report)
        systemic_tags = {"timezone", "caching", "sso", "scheduled-report", "security", "database"}
        tag_to_tickets: Dict[str, List[str]] = defaultdict(list)
        for t in self.tickets:
            for tag in t.get("tags", []):
                if tag in systemic_tags:
                    tag_to_tickets[tag].append(t["ticket_id"])

        for tag, t_ids in tag_to_tickets.items():
            for i in range(len(t_ids)):
                for j in range(i + 1, len(t_ids)):
                    a, b = t_ids[i], t_ids[j]
                    self._add_graph_edge(a, b, f"Shared systemic root-cause tag: #{tag}")
                    self._add_graph_edge(b, a, f"Shared systemic root-cause tag: #{tag}")

    def _add_graph_edge(self, src_id: str, dst_id: str, reason: str) -> None:
        existing = {edge["ticket_id"] for edge in self.root_cause_graph[src_id]}
        if dst_id not in existing and dst_id in self.all_tickets_by_id:
            dst = self.all_tickets_by_id[dst_id]
            self.root_cause_graph[src_id].append({
                "ticket_id": dst_id,
                "subject": dst.get("subject", ""),
                "category": dst.get("category", ""),
                "reason": reason,
            })

    def add_resolved_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically add or update a resolved ticket in the live index (Knowledge Flywheel)."""
        ticket = dict(ticket)
        ticket["status"] = "resolved"
        t_id = ticket["ticket_id"]
        self.all_tickets_by_id[t_id] = ticket
        self.tickets = [t for t in self.tickets if t["ticket_id"] != t_id] + [ticket]
        self._build_index()
        return ticket

    def _idf(self, token: str) -> float:
        n_docs = len(self.tickets)
        df = self.df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def predict_category(self, subject: str, description: str) -> List[Tuple[str, float]]:
        """Predict the most likely support categories for an untagged incoming ticket."""
        query_tokens = normalize_tokens(f"{subject} {description}", include_concepts=True)
        scores: Dict[str, float] = defaultdict(float)
        for tok in query_tokens:
            idf = max(self._idf(tok), 0.4)
            for cat, counter in self.category_token_weights.items():
                if tok in counter:
                    scores[cat] += idf * math.log1p(counter[tok])

        if not scores:
            return [("sync", 0.5)]

        total = sum(scores.values()) or 1.0
        ranked = sorted(
            [(cat, round(val / total, 3)) for cat, val in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def retrieve(
        self,
        subject: str,
        description: str,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute multi-lane retrieval (Lexical BM25 + Semantic Concept/N-gram + Facet Alignment + RRF).
        Returns detailed query diagnostics and ranked candidate matches.
        """
        query_text = f"{subject} {description}"
        # Weight subject tokens higher in the incoming query as well
        query_tokens_subj = normalize_tokens(subject, include_concepts=True)
        query_tokens_desc = normalize_tokens(description, include_concepts=True)
        query_counts: Counter[str] = Counter()
        for t in query_tokens_subj:
            query_counts[t] += 2.2
        for t in query_tokens_desc:
            query_counts[t] += 1.0

        query_ngrams = char_ngrams(query_text, n=4)
        query_facets = extract_facets(subject, description, self.corpus_vocab)
        predicted_categories = self.predict_category(subject, description)
        top_category = predicted_categories[0][0] if predicted_categories else "unknown"

        # Compute max theoretical BM25 for this query to calibrate scores into [0, 1]
        k1 = 1.5
        b = 0.75
        max_possible_bm25 = sum(
            q_weight * self._idf(tok) * (k1 + 1.0)
            for tok, q_weight in query_counts.items()
            if self._idf(tok) > 0
        ) or 1.0

        lexical_scores: List[float] = []
        semantic_scores: List[float] = []
        shared_terms_per_doc: List[List[str]] = []

        q_concepts = set(query_facets["concepts"])
        q_where = set(query_facets["where_facets"])
        q_what = set(query_facets["what_facets"])

        for idx, ticket in enumerate(self.tickets):
            doc_counter = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]

            # Lane 1: Weighted BM25
            bm25 = 0.0
            matched_terms: List[str] = []
            for tok, q_weight in query_counts.items():
                if tok in doc_counter:
                    tf = doc_counter[tok]
                    idf = self._idf(tok)
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    bm25 += q_weight * idf * ((tf * (k1 + 1.0)) / denom)
                    if not tok.startswith("concept_"):
                        matched_terms.append(tok)

            norm_bm25 = min(1.0, bm25 / (max_possible_bm25 * 0.62))
            lexical_scores.append(norm_bm25)
            shared_terms_per_doc.append(matched_terms)

            # Lane 2: Semantic Concept + Character 4-Gram Subword Similarity
            doc_facets = self.doc_facets[idx]
            d_concepts = set(doc_facets["concepts"])
            concept_sim = 0.0
            if q_concepts and d_concepts:
                concept_sim = len(q_concepts & d_concepts) / len(q_concepts | d_concepts)

            d_ngrams = self.doc_ngrams[idx]
            ngram_sim = 0.0
            if query_ngrams and d_ngrams:
                inter = len(query_ngrams & d_ngrams)
                ngram_sim = inter / math.sqrt(len(query_ngrams) * len(d_ngrams))

            sem_score = min(1.0, 0.60 * concept_sim + 0.40 * min(1.0, ngram_sim * 2.2))
            semantic_scores.append(sem_score)

        # Compute RRF (Reciprocal Rank Fusion) across Lexical and Semantic lanes
        lex_ranks = self._ranks(lexical_scores)
        sem_ranks = self._ranks(semantic_scores)

        candidates: List[Dict[str, Any]] = []
        for idx, ticket in enumerate(self.tickets):
            doc_facets = self.doc_facets[idx]
            d_where = set(doc_facets["where_facets"])
            d_what = set(doc_facets["what_facets"])

            where_shared = sorted(q_where & d_where)
            what_shared = sorted(q_what & d_what)
            where_match = len(where_shared) > 0
            what_match = len(what_shared) > 0

            # Determine Facet Alignment Type & Multiplier
            symptom_divergent = False
            if where_match and what_match:
                alignment_type = "Full Alignment (Feature + Symptom)"
                facet_bonus = 0.22
            elif where_match and not what_match:
                alignment_type = "Feature Match (Different Symptom)"
                symptom_divergent = True
                facet_bonus = 0.04
            elif what_match and not where_match:
                alignment_type = "Symptom Match (Different Feature)"
                symptom_divergent = True
                facet_bonus = 0.03
            else:
                alignment_type = "Category / Keyword Analog"
                facet_bonus = -0.05

            # Category alignment bonus
            cat_bonus = 0.05 if ticket.get("category") == top_category else 0.0

            # Reciprocal Rank Fusion component
            rrf = (1.0 / (10 + lex_ranks[idx])) + (1.0 / (10 + sem_ranks[idx]))
            rrf_norm = rrf / (2.0 / 11.0)  # [0, 1]

            raw_combined = (
                0.50 * lexical_scores[idx]
                + 0.25 * semantic_scores[idx]
                + 0.15 * rrf_norm
                + facet_bonus
                + cat_bonus
            )

            # Penalty if query contains an unseen third-party entity (e.g., Snowflake) not in the candidate
            if query_facets["unseen_entities"]:
                raw_combined *= 0.52
                if alignment_type == "Full Alignment (Feature + Symptom)":
                    alignment_type = "Category Analog (Unseen Integration Entity)"

            # Penalty if candidate matches the feature (WHERE) but diverges on the symptom (WHAT)
            if symptom_divergent and where_match and not what_match:
                raw_combined = min(raw_combined, 0.72)

            final_score = round(max(0.0, min(0.98, raw_combined)), 3)

            why_matched = self._build_why_matched(
                ticket=ticket,
                matched_terms=shared_terms_per_doc[idx],
                where_shared=where_shared,
                what_shared=what_shared,
                alignment_type=alignment_type,
                unseen_entities=query_facets["unseen_entities"],
            )

            t_id = ticket["ticket_id"]
            candidates.append({
                "ticket_id": t_id,
                "subject": ticket.get("subject", ""),
                "description": ticket.get("description", ""),
                "resolution_notes": ticket.get("resolution_notes", ""),
                "category": ticket.get("category", ""),
                "category_label": CATEGORY_LABELS.get(ticket.get("category", ""), ticket.get("category", "")),
                "tags": ticket.get("tags", []),
                "created_at": ticket.get("created_at", ""),
                "status": ticket.get("status", "resolved"),
                "score": final_score,
                "score_pct": int(round(final_score * 100)),
                "lexical_score": round(lexical_scores[idx], 3),
                "semantic_score": round(semantic_scores[idx], 3),
                "alignment_type": alignment_type,
                "where_shared": where_shared,
                "what_shared": what_shared,
                "symptom_divergent": symptom_divergent,
                "matched_keywords": shared_terms_per_doc[idx][:6],
                "why_matched": why_matched,
                "linked_tickets": self.root_cause_graph.get(t_id, []),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return {
            "query_facets": query_facets,
            "predicted_categories": predicted_categories,
            "top_category": top_category,
            "top_category_label": CATEGORY_LABELS.get(top_category, top_category),
            "candidates": candidates[:top_n],
        }

    @staticmethod
    def _ranks(scores: List[float]) -> List[int]:
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranks = [0] * len(scores)
        for rank_idx, (doc_idx, _) in enumerate(indexed, start=1):
            ranks[doc_idx] = rank_idx
        return ranks

    @staticmethod
    def _build_why_matched(
        ticket: Dict[str, Any],
        matched_terms: List[str],
        where_shared: List[str],
        what_shared: List[str],
        alignment_type: str,
        unseen_entities: List[str],
    ) -> str:
        parts: List[str] = []
        if where_shared and what_shared:
            parts.append(
                f"Matched feature ({', '.join(where_shared)}) + symptom ({', '.join(what_shared)})"
            )
        elif where_shared and not what_shared:
            parts.append(
                f"Same feature ({', '.join(where_shared)}), but different symptom — verify failure mode"
            )
        elif what_shared and not where_shared:
            parts.append(
                f"Shared symptom pattern ({', '.join(what_shared)}) across {ticket.get('category', '')} workflow"
            )
        elif unseen_entities:
            parts.append(
                f"Closest {ticket.get('category', '')} integration playbook (unseen entity: {', '.join(unseen_entities)})"
            )

        if matched_terms:
            parts.append(f"Shared terms: {', '.join(matched_terms[:5])}")
        return " • ".join(parts) if parts else alignment_type
