"""
Automated Test Suite for the Solari Support Ticket Similarity Assistant (TRV Engine).

Run with:
  python3 tests/test_assistant.py -v
  or:
  python3 -m unittest tests/test_assistant.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from assistant.preprocessor import normalize_tokens
from assistant.triage import SupportAssistantEngine
from starter_kit import ticket_utils

TICKETS_JSON = os.path.join(ROOT_DIR, "starter_kit", "data", "tickets.json")


class TestSupportTicketAssistant(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SupportAssistantEngine(TICKETS_JSON)

    def test_contraction_tokenization_fix(self) -> None:
        """Verify contractions like \"can't\" and \"hasn't\" do not produce spurious 't' tokens."""
        tokens = normalize_tokens("Can't log in, email hasn't shown up", include_concepts=False)
        self.assertNotIn("t", tokens)
        self.assertIn("log", tokens)
        self.assertIn("email", tokens)

    def test_nt2_blank_revenue_chart_is_direct_match(self) -> None:
        """NT-2 ('Chart shows nothing for the revenue tab') should hit Tier 1 Direct Match -> T-1201."""
        res = self.engine.analyze_ticket(
            subject="Chart shows nothing for the revenue tab",
            description="The revenue tab on the main dashboard is completely empty for this customer, no bars, no error message, just white space.",
            ticket_id="NT-2",
        )
        self.assertEqual(res["tier_code"], "DIRECT_MATCH")
        self.assertEqual(res["matches"][0]["ticket_id"], "T-1201")
        self.assertGreaterEqual(res["top_score"], 0.74)
        self.assertTrue(len(res["recommended_action_bullets"]) >= 3)

    def test_nt3_duplicate_charge_is_direct_match(self) -> None:
        """NT-3 ('Getting charged twice this month') should hit Tier 1 Direct Match -> T-1303."""
        res = self.engine.analyze_ticket(
            subject="Getting charged twice this month",
            description="Customer's card statement shows two charges for what should be a single monthly subscription payment.",
            ticket_id="NT-3",
        )
        self.assertEqual(res["tier_code"], "DIRECT_MATCH")
        self.assertEqual(res["matches"][0]["ticket_id"], "T-1303")
        self.assertEqual(res["predicted_category"], "billing")
        self.assertGreaterEqual(res["top_score"], 0.74)

    def test_nt1_password_reset_email_is_partial_match_with_symptom_warning(self) -> None:
        """
        NT-1 ('Can't log in, password reset email never arrives') matches T-1001 on feature
        (password reset) but diverges on symptom (delivery failure vs password rejection).
        Should trigger PARTIAL_MATCH and surface both T-1001 and delivery-failure tickets.
        """
        res = self.engine.analyze_ticket(
            subject="Can't log in, password reset email never arrives",
            description="A customer requested a password reset an hour ago and says the email still hasn't shown up, even after checking spam.",
            ticket_id="NT-1",
        )
        self.assertEqual(res["tier_code"], "PARTIAL_MATCH")
        self.assertEqual(res["matches"][0]["ticket_id"], "T-1001")
        self.assertTrue(res["matches"][0]["symptom_divergent"])
        matched_ids = [m["ticket_id"] for m in res["matches"]]
        self.assertTrue("T-1003" in matched_ids or "T-1403" in matched_ids)

    def test_nt4_snowflake_triggers_novel_abstention(self) -> None:
        """
        NT-4 ('Our data warehouse integration with Snowflake is failing') introduces an unseen
        integration entity ('snowflake') and must trigger Tier 3 NOVEL_ABSTENTION.
        """
        res = self.engine.analyze_ticket(
            subject="Our data warehouse integration with Snowflake is failing",
            description="Customer set up a new integration to sync data into their Snowflake warehouse and it's failing with an unclear connection error.",
            ticket_id="NT-4",
        )
        self.assertEqual(res["tier_code"], "NOVEL_ABSTENTION")
        self.assertIn("snowflake", res["facets"]["unseen_entities"])
        self.assertEqual(res["predicted_category"], "sync")

    def test_cross_ticket_root_cause_graph_links_t1401_and_t1104(self) -> None:
        """Verify T-1401 and T-1104 are bidirectionally linked in the Root-Cause Graph."""
        t1401 = self.engine.get_ticket_by_id("T-1401")
        self.assertIsNotNone(t1401)
        linked_ids = [lt["ticket_id"] for lt in t1401["linked_tickets"]]
        self.assertIn("T-1104", linked_ids)

    def test_knowledge_flywheel_promotes_novel_ticket_to_direct_match(self) -> None:
        """Promoting a resolution for NT-4 (Snowflake) immediately makes future Snowflake tickets a Direct Match."""
        promoted = self.engine.promote_new_resolution(
            subject="Our data warehouse integration with Snowflake is failing",
            description="Customer set up a new integration to sync data into their Snowflake warehouse and it's failing with an unclear connection error.",
            resolution_notes="Snowflake key-pair authentication required unencrypted PKCS#8 formatting in the connection settings. Updated connector validation.",
        )
        self.assertEqual(promoted["status"], "resolved")

        # Re-querying a Snowflake failure now hits Tier 1 DIRECT_MATCH!
        res = self.engine.analyze_ticket(
            subject="Snowflake data warehouse integration connection failing",
            description="Customer cannot connect their Snowflake warehouse sync due to a connection error.",
            ticket_id="NT-4-FOLLOWUP",
        )
        self.assertEqual(res["tier_code"], "DIRECT_MATCH")
        self.assertEqual(res["matches"][0]["ticket_id"], promoted["ticket_id"])

    def test_starter_kit_ticket_utils_backward_compatibility(self) -> None:
        """Verify starter_kit/ticket_utils.py preserves original function signatures."""
        tickets = ticket_utils.load_tickets(TICKETS_JSON)
        self.assertEqual(len(tickets), 40)
        top = ticket_utils.find_top_matches("Charged twice for monthly subscription", tickets, top_n=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0][0]["ticket_id"], "T-1303")


if __name__ == "__main__":
    unittest.main()
