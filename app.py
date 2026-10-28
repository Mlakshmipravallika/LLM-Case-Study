#!/usr/bin/env python3
"""
Interactive Web Application & REST API for the Solari Support Ticket Similarity Assistant.

Features:
1. Ticketing Board (Home View): Search by bug ID/title, filter by Status (All / New / Resolved),
   Category, or Assignee; clean rows showing Bug ID, Title, Status, Category, and Assignee.
2. Bug Detail View: Displays the reference JSON fields for the selected bug, with a floating
   circular "✨ AI Suggest" button in the top-right corner.
3. Slide-Out AI Panel (Right of Bug):
   - Unified card with Match Tier + Top Similarity + Clickable Mini Similarity Boxes (e.g., T-1001 72%,
     T-1003 48%) + Step-by-Step Recommended Agent Action Bullets.
   - Clicking any mini similarity box displays that bug's brief detail & How It Was Fixed
     (resolution_notes) below.
   - Clicking the brief detail card routes directly to that specific bug's full detail view.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List
from urllib.parse import urlparse

from assistant.triage import SupportAssistantEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKETS_PATH = os.path.join(BASE_DIR, "starter_kit", "data", "tickets.json")
NEW_TICKETS_PATH = os.path.join(BASE_DIR, "starter_kit", "data", "new_tickets_example.json")

ENGINE = SupportAssistantEngine(TICKETS_PATH)
with open(NEW_TICKETS_PATH, encoding="utf-8") as _f:
    EXAMPLE_NEW_TICKETS: List[Dict[str, Any]] = json.load(_f).get("new_tickets", [])

# Deterministic realistic assignees for board filtering
CATEGORY_ASSIGNEES = {
    "auth": "Alex Rivera",
    "sync": "Priya Patel",
    "render": "Jordan Lee",
    "billing": "Taylor Kim",
    "export": "Morgan Chen",
    "perm": "Alex Rivera",
    "perf": "Casey Zhang",
    "api": "Priya Patel",
}


def enrich_board_tickets() -> List[Dict[str, Any]]:
    """Return all tickets (New/Open + Historical Resolved) formatted for the Ticketing Board."""
    board: List[Dict[str, Any]] = []
    # 1. Incoming Open/New tickets from new_tickets_example.json
    new_assignees = ["Unassigned", "Alex Rivera", "Taylor Kim", "Priya Patel"]
    for idx, nt in enumerate(EXAMPLE_NEW_TICKETS):
        preds = ENGINE.retriever.predict_category(nt.get("subject", ""), nt.get("description", ""))
        pred_cat = preds[0][0] if preds else "sync"
        board.append({
            "ticket_id": nt["ticket_id"],
            "subject": nt.get("subject", ""),
            "description": nt.get("description", ""),
            "status": "open",
            "category": pred_cat,
            "tags": ["incoming", pred_cat],
            "created_at": "2025-12-02 (Today)",
            "resolution_notes": "",
            "assignee": new_assignees[idx % len(new_assignees)],
            "is_incoming": True,
        })

    # 2. Historical Resolved tickets from tickets.json
    for ht in reversed(ENGINE.list_historical_tickets()):
        cat = ht.get("category", "sync")
        board.append({
            "ticket_id": ht["ticket_id"],
            "subject": ht.get("subject", ""),
            "description": ht.get("description", ""),
            "status": ht.get("status", "resolved"),
            "category": cat,
            "tags": ht.get("tags", []),
            "created_at": ht.get("created_at", ""),
            "resolution_notes": ht.get("resolution_notes", ""),
            "assignee": CATEGORY_ASSIGNEES.get(cat, "Jordan Lee"),
            "is_incoming": False,
        })
    return board


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Solari Analytics — Ticketing Board &amp; AI Similarity Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    @keyframes floatPulse {
      0%, 100% { transform: translateY(0px); box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.45); }
      50% { transform: translateY(-3px); box-shadow: 0 12px 24px -4px rgba(79, 70, 229, 0.65); }
    }
    .ai-fab { animation: floatPulse 2.6s ease-in-out infinite; }
  </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen">

  <!-- Top Navigation Header -->
  <header class="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-sm">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3 cursor-pointer" onclick="showBoardView()">
        <div class="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center font-extrabold text-white shadow-sm">S</div>
        <div>
          <h1 class="font-bold text-base tracking-tight text-slate-900">Solari Support Ticketing System</h1>
          <p class="text-xs text-slate-500">Support Ticket Similarity Assistant • <span id="header-total-count" class="font-semibold text-slate-700">44</span> Total Bugs</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <button onclick="openNewTicketModal()" class="text-xs font-semibold px-3 py-1.5 rounded-lg border border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition">
          + Report New Bug
        </button>
        <div id="breadcrumb" class="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
          📋 Ticketing Board
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 py-6">

    <!-- ===================================================================== -->
    <!-- VIEW 1: TICKETING BOARD (Search + Filters + Clean Bug List)           -->
    <!-- ===================================================================== -->
    <section id="board-view" class="space-y-4">
      <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 class="text-xl font-bold text-slate-900">Ticketing Board</h2>
            <p class="text-xs text-slate-500 mt-0.5">Click any bug row below to inspect its details and launch the AI Similarity Assistant.</p>
          </div>
          <!-- Quick Status Pills -->
          <div class="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button onclick="setStatusFilter('all')" id="pill-all" class="text-xs font-semibold px-3 py-1.5 rounded-md transition">All Bugs</button>
            <button onclick="setStatusFilter('open')" id="pill-open" class="text-xs font-semibold px-3 py-1.5 rounded-md transition">● Open / New</button>
            <button onclick="setStatusFilter('resolved')" id="pill-resolved" class="text-xs font-semibold px-3 py-1.5 rounded-md transition">✓ Resolved</button>
          </div>
        </div>

        <!-- Search Bar + Filter Dropdowns -->
        <div class="grid grid-cols-1 md:grid-cols-12 gap-3">
          <div class="md:col-span-6 relative">
            <input
              id="search-input"
              type="text"
              oninput="renderBoardTable()"
              placeholder="🔍 Search by Bug ID (e.g. NT-1, T-1303) or Bug Title (e.g. Snowflake, password, chart)..."
              class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-2 text-xs text-slate-900 focus:bg-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            />
          </div>
          <div class="md:col-span-3">
            <select id="filter-category" onchange="renderBoardTable()" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-700 font-medium focus:bg-white focus:outline-none focus:border-indigo-500">
              <option value="all">Category: All Categories</option>
              <option value="auth">auth (Authentication)</option>
              <option value="sync">sync (Data Sync &amp; Integrations)</option>
              <option value="render">render (Dashboard Rendering)</option>
              <option value="billing">billing (Billing &amp; Payments)</option>
              <option value="export">export (Reports &amp; Exports)</option>
              <option value="perm">perm (Permissions &amp; Roles)</option>
              <option value="perf">perf (Performance)</option>
              <option value="api">api (API &amp; Webhooks)</option>
            </select>
          </div>
          <div class="md:col-span-3">
            <select id="filter-assignee" onchange="renderBoardTable()" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-700 font-medium focus:bg-white focus:outline-none focus:border-indigo-500">
              <option value="all">Assignee: All Assignees</option>
              <option value="Unassigned">Unassigned</option>
              <option value="Alex Rivera">Alex Rivera</option>
              <option value="Priya Patel">Priya Patel</option>
              <option value="Jordan Lee">Jordan Lee</option>
              <option value="Taylor Kim">Taylor Kim</option>
              <option value="Morgan Chen">Morgan Chen</option>
              <option value="Casey Zhang">Casey Zhang</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Clean Ticket Table (Bug ID, Title, Status, Assignee) -->
      <div class="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div class="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-500">
          <span>Showing <span id="filtered-count" class="text-slate-800">0</span> Bugs</span>
          <span>Click any row to open bug</span>
        </div>
        <div class="divide-y divide-slate-200" id="board-rows"></div>
      </div>
    </section>

    <!-- ===================================================================== -->
    <!-- VIEW 2: SPECIFIC BUG DETAIL VIEW + HOVERING AI SUGGEST CIRCLE         -->
    <!-- ===================================================================== -->
    <section id="bug-detail-view" class="hidden">
      <!-- Top Back Bar -->
      <div class="flex items-center justify-between mb-4">
        <button onclick="showBoardView()" class="text-xs font-semibold px-3.5 py-2 rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-100 transition flex items-center gap-1.5 shadow-sm">
          ← Back to Ticketing Board
        </button>
        <div id="bug-nav-hint" class="text-xs text-slate-500">
          Click the glowing <strong class="text-indigo-600">✨ AI Suggest</strong> circle on the bug card to view similar solved tickets.
        </div>
      </div>

      <!-- Dynamic Grid: Expands from Centered Single Card to 2-Column Split when AI Suggest is clicked -->
      <div id="bug-layout-grid" class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">

        <!-- LEFT CARD: The Selected Bug Details (Reference JSON Fields) -->
        <div id="bug-card-col" class="lg:col-span-12 bg-white border border-slate-200 rounded-xl p-6 shadow-sm relative transition-all">
          
          <!-- HOVERING CIRCLE "AI SUGGEST" BUTTON IN TOP-RIGHT CORNER -->
          <button
            onclick="toggleAiPanel()"
            id="ai-suggest-btn"
            title="Click to open AI Similarity Assistant"
            class="ai-fab absolute -top-3 right-5 px-4 py-2.5 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs flex items-center gap-2 border-2 border-white cursor-pointer transition"
          >
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-300 animate-ping"></span>
            <span>✨ AI Suggest</span>
          </button>

          <!-- Bug Header Metadata -->
          <div class="flex flex-wrap items-center gap-2 mb-3 pr-36">
            <span id="detail-bug-id" class="text-sm font-extrabold px-2.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200"></span>
            <span id="detail-status-badge" class="text-xs font-bold px-2.5 py-0.5 rounded border"></span>
            <span id="detail-category-badge" class="text-xs font-semibold px-2.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200"></span>
            <span id="detail-assignee-badge" class="text-xs text-slate-500 font-medium"></span>
          </div>

          <h2 id="detail-subject" class="text-xl font-bold text-slate-900 mb-4"></h2>

          <!-- Reference JSON Details Box -->
          <div class="space-y-4">
            <div class="p-4 rounded-xl bg-slate-50 border border-slate-200">
              <div class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1.5">Description (from Reference JSON)</div>
              <p id="detail-description" class="text-sm text-slate-800 leading-relaxed"></p>
            </div>

            <!-- Shown if the opened bug is ALREADY resolved (e.g., after routing to T-1001 or T-1303) -->
            <div id="detail-existing-resolution-box" class="hidden p-4 rounded-xl bg-emerald-50/70 border border-emerald-200">
              <div class="text-xs font-bold uppercase tracking-wider text-emerald-800 mb-1.5">🛠️ Resolution Notes (How This Bug Was Fixed)</div>
              <p id="detail-existing-resolution" class="text-sm text-slate-900 leading-relaxed font-medium"></p>
            </div>

            <!-- Reference JSON Metadata Table -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 text-xs">
              <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <div class="text-slate-400 font-semibold uppercase">ticket_id</div>
                <div id="meta-id" class="font-bold text-slate-800 mt-0.5"></div>
              </div>
              <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <div class="text-slate-400 font-semibold uppercase">status</div>
                <div id="meta-status" class="font-bold text-slate-800 mt-0.5"></div>
              </div>
              <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <div class="text-slate-400 font-semibold uppercase">category</div>
                <div id="meta-cat" class="font-bold text-slate-800 mt-0.5"></div>
              </div>
              <div class="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                <div class="text-slate-400 font-semibold uppercase">created_at</div>
                <div id="meta-date" class="font-bold text-slate-800 mt-0.5"></div>
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-1.5 pt-1" id="detail-tags-list"></div>

            <!-- Optional Resolve Box for Open Tickets -->
            <div id="resolve-open-bug-box" class="hidden pt-4 border-t border-slate-200">
              <div class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1.5">Close &amp; Resolve This Open Bug</div>
              <div class="flex flex-col sm:flex-row gap-2">
                <input id="new-resolution-input" type="text" placeholder="Enter resolution notes to mark this bug Resolved..." class="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-900 focus:bg-white focus:outline-none focus:border-indigo-500" />
                <button onclick="resolveCurrentOpenBug()" class="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs shrink-0 shadow-sm transition">
                  ✓ Mark Resolved &amp; Index
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- ================================================================= -->
        <!-- RIGHT PANEL: AI SUGGESTION PANEL (Revealed when AI Suggest clicked)-->
        <!-- ================================================================= -->
        <div id="ai-side-panel" class="hidden lg:col-span-7 space-y-4">
          
          <!-- UNIFIED SINGLE BOX (Your Screenshot #1: Match Tier + Mini Similarity Boxes + Action Bullets) -->
          <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
            
            <!-- Top Sub-Banner: Match Tier + Summary + Similarity Boxes Row -->
            <div id="ai-tier-box" class="rounded-xl p-4 border space-y-3">
              <div class="flex items-center justify-between gap-2">
                <div class="flex items-center gap-2">
                  <span class="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">1. Match Tier</span>
                  <span id="ai-tier-badge" class="font-bold text-sm text-slate-900"></span>
                </div>
                <button onclick="toggleAiPanel()" class="text-xs text-slate-500 hover:text-slate-800 font-semibold">✕ Hide AI</button>
              </div>

              <p id="ai-tier-summary" class="text-xs text-slate-700 leading-relaxed"></p>

              <!-- ROW OF SIMILARITY BOXES (Top Similarity + Clickable Bug Boxes from Screenshot #1) -->
              <div>
                <div class="text-xs font-semibold text-slate-600 mb-1.5">
                  👇 Click any similar bug box below to preview how it was fixed:
                </div>
                <div class="flex flex-wrap items-stretch gap-2.5">
                  <!-- Primary Top Similarity Counter Box -->
                  <div class="bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-center shadow-sm flex flex-col justify-center min-w-[105px]">
                    <div class="text-xs text-slate-500 font-medium">Top Similarity</div>
                    <div id="ai-top-score" class="text-lg font-extrabold text-slate-900"></div>
                  </div>

                  <!-- Dynamic Clickable Similarity Boxes (The 3 Red Squares in Screenshot #1) -->
                  <div id="ai-similarity-boxes" class="flex flex-wrap gap-2 flex-1"></div>
                </div>
              </div>
            </div>

            <!-- Recommended Agent Action Bullets Inside the Same Unified Box -->
            <div class="p-4 rounded-xl bg-slate-50 border border-slate-200">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  2. Recommended Agent Action (Step-by-Step Bullets)
                </span>
                <span class="text-xs text-slate-500 font-medium">Synthesized Triage Playbook</span>
              </div>
              <ul id="ai-action-bullets" class="list-disc list-inside space-y-1.5 text-xs sm:text-sm text-slate-800 leading-relaxed"></ul>
            </div>

          </div>

          <!-- SELECTED SIMILAR BUG DETAIL BOX (Your Screenshot #2: Shown at bottom when a similarity box is clicked, clicks route to that bug!) -->
          <div id="selected-match-card-wrapper" class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-bold uppercase tracking-wider text-slate-500">
                Selected Similar Bug Preview (Click Card to Route to Full Bug)
              </span>
              <span class="text-xs font-semibold text-indigo-600">👆 Click card to open bug</span>
            </div>
            <div id="selected-match-card"></div>
          </div>

        </div>

      </div>
    </section>

    <!-- Modal to Create / Test a Custom New Bug on the Board -->
    <div id="new-bug-modal" class="hidden fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 flex items-center justify-center p-4">
      <div class="bg-white border border-slate-200 rounded-2xl max-w-lg w-full p-5 shadow-xl space-y-4">
        <div class="flex items-center justify-between border-b border-slate-200 pb-3">
          <h3 class="font-bold text-base text-slate-900">Report New Incoming Bug</h3>
          <button onclick="closeNewTicketModal()" class="text-xs text-slate-500 hover:text-slate-800 font-bold">✕</button>
        </div>
        <div class="space-y-3">
          <div>
            <label class="block text-xs font-semibold text-slate-700 mb-1">Bug Title / Subject</label>
            <input id="modal-subject" type="text" placeholder="e.g., Okta SSO login redirects to a blank screen" class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-indigo-500" />
          </div>
          <div>
            <label class="block text-xs font-semibold text-slate-700 mb-1">Customer Description</label>
            <textarea id="modal-desc" rows="3" placeholder="Describe the issue..." class="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:bg-white focus:outline-none focus:border-indigo-500"></textarea>
          </div>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button onclick="closeNewTicketModal()" class="px-3.5 py-2 rounded-lg border border-slate-300 text-xs font-semibold text-slate-700 hover:bg-slate-100">Cancel</button>
          <button onclick="createNewBugOnBoard()" class="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-sm">Add to Board &amp; Open Bug</button>
        </div>
      </div>
    </div>

  </main>

  <script>
    let allBoardTickets = [];
    let statusFilter = 'all';
    let currentBug = null;
    let currentAnalysis = null;
    let aiPanelOpen = false;
    let selectedMatchIdx = 0;

    async function init() {
      await loadBoardTickets();
      setStatusFilter('all');
    }

    async function loadBoardTickets() {
      const res = await fetch('/api/board');
      const data = await res.json();
      allBoardTickets = data.tickets || [];
      document.getElementById('header-total-count').textContent = allBoardTickets.length;
    }

    function setStatusFilter(status) {
      statusFilter = status;
      const openCount = allBoardTickets.filter(t => t.status === 'open').length;
      const resolvedCount = allBoardTickets.filter(t => t.status === 'resolved').length;

      const pills = [
        { id: 'pill-all', key: 'all', label: `All Bugs (${allBoardTickets.length})` },
        { id: 'pill-open', key: 'open', label: `● Open / New (${openCount})` },
        { id: 'pill-resolved', key: 'resolved', label: `✓ Resolved (${resolvedCount})` },
      ];
      pills.forEach(p => {
        const el = document.getElementById(p.id);
        el.textContent = p.label;
        if (p.key === statusFilter) {
          el.className = 'text-xs font-bold px-3 py-1.5 rounded-md bg-white text-indigo-700 shadow-xs border border-slate-200 transition';
        } else {
          el.className = 'text-xs font-medium px-3 py-1.5 rounded-md text-slate-600 hover:text-slate-900 transition';
        }
      });
      renderBoardTable();
    }

    function renderBoardTable() {
      const q = (document.getElementById('search-input').value || '').trim().toLowerCase();
      const catFilter = document.getElementById('filter-category').value;
      const assigneeFilter = document.getElementById('filter-assignee').value;

      const filtered = allBoardTickets.filter(t => {
        if (statusFilter !== 'all' && t.status !== statusFilter) return false;
        if (catFilter !== 'all' && t.category !== catFilter) return false;
        if (assigneeFilter !== 'all' && t.assignee !== assigneeFilter) return false;
        if (q) {
          const hay = `${t.ticket_id} ${t.subject} ${t.category} ${t.assignee}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      document.getElementById('filtered-count').textContent = filtered.length;
      const rowsEl = document.getElementById('board-rows');

      if (filtered.length === 0) {
        rowsEl.innerHTML = `<div class="p-8 text-center text-xs text-slate-500">No bugs match the current filters.</div>`;
        return;
      }

      rowsEl.innerHTML = filtered.map(t => {
        const isOpen = t.status === 'open';
        const statusBadge = isOpen
          ? `<span class="text-xs font-bold px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-300">● Open / New</span>`
          : `<span class="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200">✓ Resolved</span>`;

        return `
          <div onclick="openBugDetail('${t.ticket_id}')" class="px-4 py-3 hover:bg-indigo-50/50 cursor-pointer transition flex flex-wrap items-center justify-between gap-3 group">
            <div class="flex items-center gap-3 min-w-0">
              <span class="text-xs font-extrabold px-2.5 py-1 rounded-md ${
                isOpen ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-800 border border-slate-300'
              } shrink-0">${t.ticket_id}</span>
              <span class="text-sm font-semibold text-slate-900 group-hover:text-indigo-700 group-hover:underline truncate">${t.subject}</span>
            </div>
            <div class="flex items-center gap-3 shrink-0">
              <span class="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200 font-medium">${t.category}</span>
              <span class="text-xs text-slate-500 hidden sm:inline">👤 ${t.assignee}</span>
              ${statusBadge}
              <span class="text-xs font-bold text-indigo-600 opacity-0 group-hover:opacity-100 transition">Open →</span>
            </div>
          </div>
        `;
      }).join('');
    }

    function showBoardView() {
      document.getElementById('bug-detail-view').classList.add('hidden');
      document.getElementById('board-view').classList.remove('hidden');
      document.getElementById('breadcrumb').textContent = '📋 Ticketing Board';
      history.replaceState(null, '', ' ');
    }

    async function openBugDetail(ticketId, autoOpenAi = false) {
      const bug = allBoardTickets.find(t => t.ticket_id === ticketId);
      if (!bug) return;
      currentBug = bug;
      aiPanelOpen = false;
      selectedMatchIdx = 0;

      // Render Bug Details (Left Card)
      document.getElementById('detail-bug-id').textContent = bug.ticket_id;
      const isOpen = bug.status === 'open';
      const statusEl = document.getElementById('detail-status-badge');
      statusEl.textContent = isOpen ? '● Status: Open / New' : '✓ Status: Resolved';
      statusEl.className = isOpen
        ? 'text-xs font-bold px-2.5 py-0.5 rounded border bg-amber-50 text-amber-800 border-amber-300'
        : 'text-xs font-bold px-2.5 py-0.5 rounded border bg-emerald-50 text-emerald-800 border-emerald-300';

      document.getElementById('detail-category-badge').textContent = `Category: ${bug.category}`;
      document.getElementById('detail-assignee-badge').textContent = `👤 Assignee: ${bug.assignee}`;
      document.getElementById('detail-subject').textContent = bug.subject;
      document.getElementById('detail-description').textContent = bug.description;

      document.getElementById('meta-id').textContent = bug.ticket_id;
      document.getElementById('meta-status').textContent = bug.status;
      document.getElementById('meta-cat').textContent = bug.category;
      document.getElementById('meta-date').textContent = bug.created_at;

      const resBox = document.getElementById('detail-existing-resolution-box');
      const resolveFormBox = document.getElementById('resolve-open-bug-box');
      if (bug.resolution_notes) {
        resBox.classList.remove('hidden');
        document.getElementById('detail-existing-resolution').textContent = bug.resolution_notes;
        resolveFormBox.classList.add('hidden');
      } else {
        resBox.classList.add('hidden');
        resolveFormBox.classList.remove('hidden');
      }

      document.getElementById('detail-tags-list').innerHTML = (bug.tags || [])
        .map(tag => `<span class="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-300 text-slate-600 font-medium">#${tag}</span>`)
        .join('');

      // Reset AI Panel to collapsed initially unless autoOpenAi is true
      setAiPanelState(false);

      document.getElementById('board-view').classList.add('hidden');
      document.getElementById('bug-detail-view').classList.remove('hidden');
      document.getElementById('breadcrumb').textContent = `📋 Board → Bug ${bug.ticket_id}`;
      window.location.hash = `bug/${bug.ticket_id}`;

      // Pre-fetch AI Similarity Analysis for this bug
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: bug.ticket_id,
          subject: bug.subject,
          description: bug.description,
          exclude_ticket_id: bug.ticket_id,
          top_n: 3
        })
      });
      currentAnalysis = await res.json();
      renderAiPanelContent();

      if (autoOpenAi) {
        setAiPanelState(true);
      }
    }

    function toggleAiPanel() {
      setAiPanelState(!aiPanelOpen);
    }

    function setAiPanelState(open) {
      aiPanelOpen = open;
      const leftCol = document.getElementById('bug-card-col');
      const rightPanel = document.getElementById('ai-side-panel');
      const aiBtn = document.getElementById('ai-suggest-btn');

      if (open) {
        leftCol.className = 'lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-sm relative transition-all';
        rightPanel.classList.remove('hidden');
        aiBtn.innerHTML = `<span>✨ AI Active</span>`;
      } else {
        leftCol.className = 'lg:col-span-12 bg-white border border-slate-200 rounded-xl p-6 shadow-sm relative transition-all';
        rightPanel.classList.add('hidden');
        aiBtn.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-emerald-300 animate-ping"></span><span>✨ AI Suggest</span>`;
      }
    }

    function renderAiPanelContent() {
      if (!currentAnalysis) return;
      const r = currentAnalysis;

      const tierBox = document.getElementById('ai-tier-box');
      let tierStyle = 'border-emerald-300 bg-emerald-50/80';
      if (r.tier_code === 'PARTIAL_MATCH') tierStyle = 'border-amber-300 bg-amber-50/80';
      if (r.tier_code === 'NOVEL_ABSTENTION') tierStyle = 'border-orange-300 bg-orange-50/80';
      tierBox.className = `rounded-xl p-4 border space-y-3 ${tierStyle}`;

      document.getElementById('ai-tier-badge').textContent = r.tier_badge;
      document.getElementById('ai-tier-summary').textContent = r.tier_summary;
      document.getElementById('ai-top-score').textContent = `${r.top_score_pct}%`;

      document.getElementById('ai-action-bullets').innerHTML = (r.recommended_action_bullets || [])
        .map(b => `<li>${b}</li>`)
        .join('');

      renderSimilarityMiniBoxes();
      renderSelectedMatchCard();
    }

    function renderSimilarityMiniBoxes() {
      const matches = (currentAnalysis && currentAnalysis.matches) || [];
      const container = document.getElementById('ai-similarity-boxes');

      container.innerHTML = matches.map((m, idx) => {
        const isSelected = idx === selectedMatchIdx;
        return `
          <button
            onclick="selectSimilarityBox(${idx})"
            class="px-3.5 py-2 rounded-lg border text-left transition flex flex-col justify-center min-w-[125px] cursor-pointer ${
              isSelected
                ? 'bg-indigo-600 text-white border-indigo-700 shadow-sm ring-2 ring-indigo-300'
                : 'bg-white text-slate-800 border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/40 shadow-xs'
            }"
          >
            <div class="flex items-center justify-between gap-2">
              <span class="text-xs font-extrabold">${m.ticket_id}</span>
              <span class="text-xs font-bold px-1.5 py-0.5 rounded ${
                isSelected ? 'bg-indigo-800 text-indigo-100' : 'bg-indigo-50 text-indigo-700'
              }">${m.score_pct}%</span>
            </div>
            <div class="text-xs truncate max-w-[140px] mt-0.5 ${isSelected ? 'text-indigo-100' : 'text-slate-500'}">
              ${m.category} • Click details
            </div>
          </button>
        `;
      }).join('');
    }

    function selectSimilarityBox(idx) {
      selectedMatchIdx = idx;
      renderSimilarityMiniBoxes();
      renderSelectedMatchCard();
    }

    function renderSelectedMatchCard() {
      const matches = (currentAnalysis && currentAnalysis.matches) || [];
      const cardEl = document.getElementById('selected-match-card');
      if (matches.length === 0) {
        cardEl.innerHTML = `<div class="text-xs text-slate-500 p-3">No similar historical tickets found.</div>`;
        return;
      }

      const m = matches[selectedMatchIdx] || matches[0];
      cardEl.innerHTML = `
        <div
          onclick="openBugDetail('${m.ticket_id}')"
          class="cursor-pointer group rounded-xl border border-slate-200 bg-slate-50/70 hover:bg-indigo-50/40 hover:border-indigo-400 p-4 transition shadow-xs"
        >
          <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div class="flex items-center gap-2">
              <span class="text-xs font-extrabold px-2.5 py-0.5 rounded bg-indigo-100 text-indigo-800 border border-indigo-200">${m.ticket_id}</span>
              <h3 class="text-sm font-bold text-slate-900 group-hover:text-indigo-700 group-hover:underline">${m.subject}</h3>
              <span class="text-xs font-bold text-indigo-600">→ Open Ticket</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span class="text-xs px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700 font-medium">${m.alignment_type}</span>
              <span class="text-xs font-bold px-2.5 py-0.5 rounded bg-indigo-600 text-white">${m.score_pct}% Match</span>
            </div>
          </div>

          <div class="text-xs text-slate-600 mb-2.5">
            <strong>Why it matched:</strong> ${m.why_matched} • <span class="text-slate-500">Resolved ${m.created_at} (${m.category})</span>
          </div>

          <div class="text-xs p-3 rounded-lg bg-white border border-slate-200 text-slate-800 leading-relaxed">
            <strong class="text-emerald-700">🛠️ How Fixed (resolution_notes):</strong> ${m.resolution_notes}
          </div>
        </div>
      `;
    }

    async function resolveCurrentOpenBug() {
      if (!currentBug) return;
      const notes = document.getElementById('new-resolution-input').value.trim();
      if (!notes) {
        alert('Please enter brief resolution notes to mark this bug Resolved.');
        return;
      }
      await fetch('/api/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticket_id: currentBug.ticket_id,
          subject: currentBug.subject,
          description: currentBug.description,
          resolution_notes: notes,
          category: currentBug.category
        })
      });
      document.getElementById('new-resolution-input').value = '';
      await loadBoardTickets();
      await openBugDetail(currentBug.ticket_id, true);
    }

    function openNewTicketModal() {
      document.getElementById('new-bug-modal').classList.remove('hidden');
    }

    function closeNewTicketModal() {
      document.getElementById('new-bug-modal').classList.add('hidden');
    }

    async function createNewBugOnBoard() {
      const subj = document.getElementById('modal-subject').value.trim();
      const desc = document.getElementById('modal-desc').value.trim();
      if (!subj) return;
      const res = await fetch('/api/create_open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject: subj, description: desc })
      });
      const created = await res.json();
      document.getElementById('modal-subject').value = '';
      document.getElementById('modal-desc').value = '';
      closeNewTicketModal();
      await loadBoardTickets();
      await openBugDetail(created.ticket_id, false);
    }

    init();
  </script>
</body>
</html>
"""


class AssistantRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving both the Ticketing Board Web UI and JSON REST API endpoints."""

    def _send_json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/board":
            self._send_json({"tickets": enrich_board_tickets()})
            return

        if path == "/api/examples":
            self._send_json({
                "examples": EXAMPLE_NEW_TICKETS,
                "historical_count": len(ENGINE.list_historical_tickets()),
            })
            return

        if path == "/api/tickets":
            self._send_json({"tickets": ENGINE.list_historical_tickets()})
            return

        if path.startswith("/api/tickets/"):
            ticket_id = path.split("/api/tickets/", 1)[1]
            ticket = ENGINE.get_ticket_by_id(ticket_id)
            if not ticket:
                self._send_json({"error": f"Ticket {ticket_id} not found"}, status=404)
                return
            self._send_json(ticket)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if parsed.path == "/api/analyze":
            exclude_id = body.get("exclude_ticket_id")
            res = ENGINE.analyze_ticket(
                subject=body.get("subject", ""),
                description=body.get("description", ""),
                ticket_id=body.get("ticket_id", "NEW"),
                top_n=int(body.get("top_n", 3)) + 1,
            )
            if exclude_id:
                res["matches"] = [m for m in res["matches"] if m["ticket_id"] != exclude_id][: int(body.get("top_n", 3))]
                if res["matches"]:
                    res["top_score"] = res["matches"][0]["score"]
                    res["top_score_pct"] = res["matches"][0]["score_pct"]
            else:
                res["matches"] = res["matches"][: int(body.get("top_n", 3))]
            self._send_json(res)
            return

        if parsed.path == "/api/resolve":
            t_id = body.get("ticket_id")
            promoted = ENGINE.promote_new_resolution(
                subject=body.get("subject", ""),
                description=body.get("description", ""),
                resolution_notes=body.get("resolution_notes", ""),
                category=body.get("category", ""),
                ticket_id=t_id,
            )
            # If it was in EXAMPLE_NEW_TICKETS, remove it from open queue since it's now in resolved index
            global EXAMPLE_NEW_TICKETS
            EXAMPLE_NEW_TICKETS = [nt for nt in EXAMPLE_NEW_TICKETS if nt.get("ticket_id") != t_id]
            self._send_json({
                "ticket": promoted,
                "historical_count": len(ENGINE.list_historical_tickets()),
            })
            return

        if parsed.path == "/api/create_open":
            new_id = f"NT-{len(EXAMPLE_NEW_TICKETS) + 5}"
            new_bug = {
                "ticket_id": new_id,
                "subject": body.get("subject", ""),
                "description": body.get("description", ""),
            }
            EXAMPLE_NEW_TICKETS.insert(0, new_bug)
            self._send_json(new_bug)
            return

        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, fmt: str, *args: object) -> None:
        return  # Keep console clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Solari Support Ticket Similarity Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), AssistantRequestHandler)
    print(f"Solari Support Ticket Similarity Assistant running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
