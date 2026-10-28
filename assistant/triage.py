"""
Stage 3 & Stage 4: 3-Tier Calibrated Confidence Gate & Actionable Playbook Synthesizer.

Evaluates retrieved candidates across three operating tiers:
- Tier 1 (DIRECT_MATCH): High-confidence alignment on both WHERE (feature) and WHAT (symptom).
- Tier 2 (PARTIAL_MATCH): Feature match with symptom divergence (or vice versa) — warns agent.
- Tier 3 (NOVEL_ABSTENTION): Unseen entity (e.g. Snowflake) or low confidence — abstains from
  claiming a direct match and generates a category-level diagnostic playbook.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from typing import Any, Dict, List

from assistant.retriever import HybridTicketRetriever


class SupportAssistantEngine:
    """End-to-end Triage-Retrieve-Verify (TRV) Support Ticket Similarity Assistant."""

    DIRECT_THRESHOLD = 0.74
    PARTIAL_THRESHOLD = 0.42
    MIN_DISPLAY_SCORE = 0.22

    def __init__(self, tickets_path: str) -> None:
        self.tickets_path = tickets_path
        with open(tickets_path, encoding="utf-8") as f:
            data = json.load(f)
        self.raw_tickets: List[Dict[str, Any]] = data.get("tickets", [])
        self.retriever = HybridTicketRetriever(self.raw_tickets, resolved_only=True)

    def get_ticket_by_id(self, ticket_id: str) -> Dict[str, Any] | None:
        """Retrieve a single historical ticket record by ID (with its root-cause graph links)."""
        t = self.retriever.all_tickets_by_id.get(ticket_id)
        if not t:
            return None
        result = dict(t)
        result["linked_tickets"] = self.retriever.root_cause_graph.get(ticket_id, [])
        return result

    def list_historical_tickets(self) -> List[Dict[str, Any]]:
        """Return all indexed historical tickets."""
        return self.retriever.tickets

    def analyze_ticket(
        self,
        subject: str,
        description: str,
        ticket_id: str = "NEW",
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """Run full Triage -> Retrieve -> Verify pipeline for an incoming support ticket."""
        retrieval = self.retriever.retrieve(subject, description, top_n=max(top_n, 5))
        candidates = retrieval["candidates"]
        q_facets = retrieval["query_facets"]
        top_cat = retrieval["top_category"]
        top_cat_label = retrieval["top_category_label"]

        top_cand = candidates[0] if candidates else None
        top_score = top_cand["score"] if top_cand else 0.0
        unseen = q_facets["unseen_entities"]

        # Stage 3: 3-Tier Calibrated Confidence Gate
        if (
            top_cand
            and not unseen
            and top_score >= self.DIRECT_THRESHOLD
            and not top_cand["symptom_divergent"]
        ):
            tier_code = "DIRECT_MATCH"
            tier_badge = "🟢 Direct Match — Known Issue Pattern"
            confidence_band = "High"
            tier_summary = (
                f"Both product feature ({', '.join(q_facets['where_facets'])}) and symptom "
                f"({', '.join(q_facets['what_facets'])}) strongly match resolved ticket {top_cand['ticket_id']}."
            )
            # Show top match + any strong secondary matches
            selected_matches = [
                c for c in candidates[:top_n]
                if c["score"] >= 0.40 or c["ticket_id"] == top_cand["ticket_id"]
            ]
        elif (
            top_cand
            and not unseen
            and top_score >= self.PARTIAL_THRESHOLD
        ):
            tier_code = "PARTIAL_MATCH"
            tier_badge = "🟡 Partial Match — Verify Symptom Difference"
            confidence_band = "Medium"
            if top_cand["symptom_divergent"]:
                tier_summary = (
                    f"{top_cand['ticket_id']} matches the feature ({', '.join(top_cand['where_shared']) or top_cat}), "
                    f"but the customer's symptom ({', '.join(q_facets['what_facets'])}) differs from {top_cand['ticket_id']}'s failure mode."
                )
            else:
                tier_summary = (
                    f"Related historical tickets found in '{top_cat_label}', but verify exact symptom alignment before applying."
                )
            selected_matches = [c for c in candidates[:top_n] if c["score"] >= self.MIN_DISPLAY_SCORE]
        else:
            tier_code = "NOVEL_ABSTENTION"
            confidence_band = "Low (Abstained — Category Fallback)"
            if unseen:
                entity_str = ", ".join(unseen)
                tier_badge = f"🟠 Novel Issue — No Past '{entity_str.title()}' Tickets"
                tier_summary = (
                    f"Unseen integration/entity detected ({entity_str}). Abstaining from false exact match; "
                    f"surfacing closest '{top_cat_label}' diagnostic playbooks for first-principles triage."
                )
            else:
                tier_badge = "🟠 Novel Issue — No Strong Historical Match"
                tier_summary = (
                    f"No historical ticket exceeded the similarity threshold. Routed to '{top_cat_label}' "
                    f"for manual triage using related category playbooks."
                )
            # Surface top 2-3 category analogs so the agent still gets diagnostic checklists
            cat_analogs = [
                c for c in candidates if c["category"] == top_cat or c["score"] >= 0.15
            ][:top_n]
            selected_matches = cat_analogs

        # Stage 4: Generate Bulleted Recommended Agent Actions
        action_bullets = self._generate_action_bullets(
            tier_code=tier_code,
            subject=subject,
            description=description,
            q_facets=q_facets,
            top_cat=top_cat,
            top_cat_label=top_cat_label,
            matches=selected_matches,
        )

        return {
            "ticket_id": ticket_id,
            "subject": subject,
            "description": description,
            "facets": q_facets,
            "predicted_category": top_cat,
            "predicted_category_label": top_cat_label,
            "category_distribution": retrieval["predicted_categories"][:3],
            "tier_code": tier_code,
            "tier_badge": tier_badge,
            "confidence_band": confidence_band,
            "top_score": top_score,
            "top_score_pct": int(round(top_score * 100)),
            "tier_summary": tier_summary,
            "recommended_action_bullets": action_bullets,
            "matches": selected_matches,
        }

    def promote_new_resolution(
        self,
        subject: str,
        description: str,
        resolution_notes: str,
        category: str = "",
        tags: List[str] | None = None,
        ticket_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Promote a newly resolved incoming ticket into the live index (Knowledge Flywheel).
        Allows a novel issue (like NT-4 Snowflake) to immediately become a Tier 1 Direct Match.
        """
        if not category:
            preds = self.retriever.predict_category(subject, description)
            category = preds[0][0] if preds else "sync"
        if not ticket_id:
            next_num = 1801 + len(self.retriever.tickets)
            ticket_id = f"T-{next_num}"
        if not tags:
            facets = self.retriever.retrieve(subject, description, top_n=1)["query_facets"]
            tags = list(dict.fromkeys(
                facets["unseen_entities"] + facets["where_facets"] + [category]
            ))[:4]

        new_record = {
            "ticket_id": ticket_id,
            "category": category,
            "subject": subject,
            "description": description,
            "resolution_notes": resolution_notes,
            "tags": tags,
            "created_at": str(date.today()),
            "status": "resolved",
        }
        return self.retriever.add_resolved_ticket(new_record)

    def _generate_action_bullets(
        self,
        tier_code: str,
        subject: str,
        description: str,
        q_facets: Dict[str, List[str]],
        top_cat: str,
        top_cat_label: str,
        matches: List[Dict[str, Any]],
    ) -> List[str]:
        """Synthesize clear, scannable bullet points for the Support Agent."""
        if not matches:
            return [
                f"Route ticket to the {top_cat_label} ({top_cat}) queue for L2 investigation.",
                "Gather customer account ID, timestamps, and reproduction screenshots.",
                "Document the resolution once solved to seed the similarity index.",
            ]

        primary = matches[0]
        p_id = primary["ticket_id"]
        res_sentences = [
            s.strip() for s in primary["resolution_notes"].split(".") if s.strip()
        ]
        root_cause_sent = res_sentences[0] + "." if res_sentences else primary["resolution_notes"]
        fix_sent = (
            ". ".join(res_sentences[1:]) + "."
            if len(res_sentences) > 1
            else primary["resolution_notes"]
        )

        if tier_code == "DIRECT_MATCH":
            bullets = [
                f"Verify Root Cause (from {p_id}): {root_cause_sent}",
                f"Apply Proven Fix: {fix_sent}",
            ]
            if primary.get("linked_tickets"):
                linked = primary["linked_tickets"][0]
                bullets.append(
                    f"Systemic Root-Cause Link: Check related ticket {linked['ticket_id']} "
                    f"({linked['subject']}) — {linked['reason']}."
                )
            bullets.append(
                f"Customer Communication: Confirm the known {primary['category']} behavior "
                f"from {p_id} and share the resolution step above for a consistent response."
            )
            return bullets

        if tier_code == "PARTIAL_MATCH":
            bullets = []
            if primary["symptom_divergent"]:
                bullets.append(
                    f"⚠️ Symptom Divergence Check: {p_id} involved the same feature "
                    f"({', '.join(primary['where_shared']) or primary['category']}), "
                    f"but its failure mode was different ({root_cause_sent}) — do not blindly copy {p_id}'s reply yet."
                )
            else:
                bullets.append(
                    f"Primary Reference ({p_id}): Check if root cause applies — {root_cause_sent}"
                )

            if len(matches) > 1:
                sec = matches[1]
                sec_sents = [s.strip() for s in sec["resolution_notes"].split(".") if s.strip()]
                sec_summary = sec_sents[0] + "." if sec_sents else sec["resolution_notes"]
                bullets.append(
                    f"Check Secondary Symptom Match ({sec['ticket_id']} - {sec['subject']}): "
                    f"{sec_summary}"
                )

            bullets.append(
                f"Follow-up Verification: Once the primary symptom ({', '.join(q_facets['what_facets'])}) "
                f"is addressed, verify the {p_id} fix ({fix_sent})"
            )
            return bullets

        # Tier 3: NOVEL_ABSTENTION
        unseen_str = ", ".join(q_facets["unseen_entities"]) if q_facets["unseen_entities"] else "this specific pattern"
        bullets = [
            f"Novel Issue Alert: No resolved historical tickets match '{unseen_str}' directly — abstaining from automatic answer.",
            f"Step 1 — Triage via {top_cat_label} Playbook ({p_id}): Check the most common {top_cat} root cause first ({root_cause_sent})",
        ]
        if len(matches) > 1:
            sec = matches[1]
            sec_sents = [s.strip() for s in sec["resolution_notes"].split(".") if s.strip()]
            sec_summary = sec_sents[0] + "." if sec_sents else sec["resolution_notes"]
            bullets.append(
                f"Step 2 — Secondary {top_cat.upper()} Check ({sec['ticket_id']}): {sec_summary}"
            )
        bullets.append(
            "Step 3 — Knowledge Flywheel: Once resolved, click 'Promote Resolution to Index' so future tickets match automatically."
        )
        return bullets
