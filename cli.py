#!/usr/bin/env python3
"""
CLI Runner for the Solari Support Ticket Similarity Assistant.

Usage examples:
  1. Run against all 4 example incoming tickets (NT-1 to NT-4):
     python3 cli.py

  2. Test a custom incoming ticket from the command line:
     python3 cli.py --subject "Okta login shows blank screen" --description "After SSO sign in, page is white"

  3. Inspect a specific historical ticket by ID (simulating ticket routing):
     python3 cli.py --inspect T-1401
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from assistant.triage import SupportAssistantEngine

DEFAULT_TICKETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "starter_kit", "data", "tickets.json"
)
DEFAULT_NEW_TICKETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "starter_kit",
    "data",
    "new_tickets_example.json",
)


def print_analysis(result: Dict[str, Any]) -> None:
    """Pretty-print the 4 core assistant sections for an incoming support ticket."""
    print("=" * 88)
    print(f"INCOMING BUG / TICKET: [{result['ticket_id']}] {result['subject']}")
    print(f"Description : {result['description']}")
    print(
        f"Auto-Facets : Category={result['predicted_category']} ({result['predicted_category_label']}) | "
        f"WHERE={', '.join(result['facets']['where_facets'])} | "
        f"WHAT={', '.join(result['facets']['what_facets'])}"
    )
    if result["facets"]["unseen_entities"]:
        print(f"Unseen Tech : {', '.join(result['facets']['unseen_entities'])}")
    print("=" * 88)

    print(f"\n[1] MATCH STATUS : {result['tier_badge']}  (Top Score: {result['top_score_pct']}%)")
    print(f"    Summary      : {result['tier_summary']}")

    print("\n[2] RECOMMENDED ACTION FOR AGENT (Bullet Points):")
    for bullet in result["recommended_action_bullets"]:
        print(f"    • {bullet}")

    print("\n[3] SIMILAR SOLVED TICKETS & [4] HOW THEY WERE FIXED (resolution_notes):")
    for idx, match in enumerate(result["matches"], start=1):
        print("    " + "-" * 80)
        print(
            f"    #{idx} [{match['ticket_id']}] {match['subject']}  "
            f"({match['score_pct']}% | {match['alignment_type']})"
        )
        print(
            f"       Category: {match['category']} | Resolved: {match['created_at']} | "
            f"Tags: {', '.join(match['tags'])}"
        )
        print(f"       Why it matched : {match['why_matched']}")
        print(f"       Customer Issue : {match['description']}")
        print(f"       🛠️  How Fixed   : {match['resolution_notes']}")
        if match.get("linked_tickets"):
            links_str = ", ".join(
                f"{lt['ticket_id']} ({lt['reason']})" for lt in match["linked_tickets"]
            )
            print(f"       🔗 Linked Bugs  : {links_str}")
    print("\n")


def print_historical_ticket(ticket: Dict[str, Any]) -> None:
    """Print full historical ticket record when routed to a specific ticket ID."""
    print("=" * 88)
    print(f"HISTORICAL TICKET RECORD: [{ticket['ticket_id']}] {ticket['subject']}")
    print(
        f"Status: {ticket.get('status', 'resolved').upper()} | "
        f"Category: {ticket.get('category')} | Created: {ticket.get('created_at')}"
    )
    print(f"Tags  : {', '.join(ticket.get('tags', []))}")
    print("-" * 88)
    print(f"Customer Description:\n  {ticket.get('description', '')}\n")
    print(f"🛠️  How It Was Fixed (resolution_notes):\n  {ticket.get('resolution_notes', '')}")
    if ticket.get("linked_tickets"):
        print("\n🔗 Cross-Referenced / Systemic Linked Tickets:")
        for lt in ticket["linked_tickets"]:
            print(f"  • [{lt['ticket_id']}] {lt['subject']} — {lt['reason']}")
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solari Support Ticket Similarity Assistant (TRV Engine)"
    )
    parser.add_argument(
        "--tickets",
        default=DEFAULT_TICKETS_PATH,
        help="Path to historical tickets JSON file",
    )
    parser.add_argument(
        "--new-tickets",
        default=DEFAULT_NEW_TICKETS_PATH,
        help="Path to incoming test tickets JSON file",
    )
    parser.add_argument("--subject", help="Custom incoming ticket subject")
    parser.add_argument("--description", default="", help="Custom incoming ticket description")
    parser.add_argument("--inspect", help="Inspect a specific historical ticket ID (e.g. T-1303)")
    parser.add_argument("--top-n", type=int, default=3, help="Number of similar matches to return")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    args = parser.parse_args()

    engine = SupportAssistantEngine(args.tickets)

    if args.inspect:
        ticket = engine.get_ticket_by_id(args.inspect)
        if not ticket:
            print(f"Ticket '{args.inspect}' not found.", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(ticket, indent=2))
        else:
            print_historical_ticket(ticket)
        return 0

    if args.subject:
        res = engine.analyze_ticket(
            subject=args.subject,
            description=args.description,
            ticket_id="CUSTOM-1",
            top_n=args.top_n,
        )
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print_analysis(res)
        return 0

    with open(args.new_tickets, encoding="utf-8") as f:
        new_tickets = json.load(f).get("new_tickets", [])

    all_results = []
    for nt in new_tickets:
        res = engine.analyze_ticket(
            subject=nt.get("subject", ""),
            description=nt.get("description", ""),
            ticket_id=nt.get("ticket_id", "NEW"),
            top_n=args.top_n,
        )
        all_results.append(res)
        if not args.json:
            print_analysis(res)

    if args.json:
        print(json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
