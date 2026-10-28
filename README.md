# Solari Analytics — Support Ticket Similarity Assistant (`v1`)

An end-to-end **Triage – Retrieve – Verify (TRV)** Support Ticketing System & AI Copilot built for Solari Analytics' Customer Support team.

## Interactive User Experience (3-Step Flow)

1. **Ticketing Board (Home View):**
   - Displays all tickets in the system (**4 Open/New incoming bugs** `NT-1`–`NT-4` + **40 Historical Resolved bugs** `T-1001`–`T-1705`, plus a **`+ Report New Bug`** modal).
   - Includes a real-time **Search Bar** (filter by Bug ID or Bug Title) and filters for **Status** (`All`, `Open / New`, `Resolved`), **Category**, and **Assignee**.
2. **Bug Detail View + Floating `✨ AI Suggest` Button:**
   - Clicking any bug row on the Ticketing Board opens that bug's full detail view (`ticket_id`, `subject`, `description`, `status`, `category`, `created_at`, `tags`, and `resolution_notes` if resolved).
   - A hovering **`✨ AI Suggest`** button sits in the top-right corner of the bug card.
3. **Slide-Out AI Panel (Right of Bug):**
   - Clicking **`✨ AI Suggest`** opens the AI Copilot panel on the right:
     - **Unified Match & Action Box:** Shows the **Calibrated Match Tier** (`🟢 Direct Match`, `🟡 Partial Match`, or `🟠 Novel Issue Abstention`), **Top Similarity %**, **Clickable Mini Similarity Boxes** for each matching past bug (e.g., `T-1001 · 72%`, `T-1003 · 48%`), and **Step-by-Step Bullet-Point Recommended Agent Actions**.
     - **Selected Similar Bug Preview & Routing:** Clicking any mini similarity box displays that bug's brief detail and **`🛠️ How Fixed (resolution_notes)`** below; clicking the preview card **routes directly to that specific bug's full detail view**.

---

## Quickstart (How to Run)

**Prerequisites:** Python 3.9+ (uses only the Python standard library — **zero external `pip` packages or API keys required**, ensuring instant offline execution inside the Percipio sandbox).

### 1. Launch the Interactive Web UI (Recommended for Live Demo)

```bash
python3 app.py --port 8000
```
*(On Windows inside the Percipio sandbox, run `python app.py`)*

Then open **`http://127.0.0.1:8000`** in your browser:
1. Browse and filter bugs on the **Ticketing Board**.
2. Click any bug (`NT-1`, `NT-2`, `NT-3`, `NT-4`) to open its detail view.
3. Click **`✨ AI Suggest`** in the top-right corner of the bug card to open the AI panel.
4. Click the **mini similarity boxes** to preview how each past bug was fixed, and click the bottom preview card to route directly to that bug.

### 2. Run the Command-Line Interface (CLI)

Evaluate all 4 test tickets in `starter_kit/data/new_tickets_example.json`:

```bash
python3 cli.py
```

Test any custom incoming bug via CLI flags:

```bash
python3 cli.py \
  --subject "CSV export is missing recent transactions" \
  --description "Exported CSV is missing the last few days of data"
```

Inspect a specific historical ticket (and its cross-referenced root-cause links, e.g., `T-1401` $\leftrightarrow$ `T-1104`):

```bash
python3 cli.py --inspect T-1401
```

### 3. Run the Automated Test Suite

```bash
python3 tests/test_assistant.py -v
```

---

## Project Structure

```text
.
├── SOLUTION_DESIGN.md               # Deliverable 1: 2-page Architecture & Trade-Off Design Doc
├── README.md                        # Deliverable 3: Setup, usage, and known limitations
├── app.py                           # Deliverable 2 (Ticketing Board + AI Suggest Web UI & REST API)
├── cli.py                           # Deliverable 2 (CLI runner for batch & custom ticket triage)
├── requirements.txt                 # Zero required external packages (pure Python 3 stdlib)
├── assistant/                       # Core Triage–Retrieve–Verify (TRV) Engine
│   ├── __init__.py
│   ├── preprocessor.py              # Contraction-safe tokenizer, stemmer & WHERE/WHAT facet extractor
│   ├── retriever.py                 # Field-weighted BM25 + Semantic Concept/4-Gram + Root-Cause Graph + RRF
│   └── triage.py                    # 3-Tier Confidence Gate & Bullet-Point Action Synthesizer
├── starter_kit/
│   ├── README.md                    # Original starter kit notes
│   ├── ticket_utils.py              # Upgraded & backward-compatible starter utility functions
│   └── data/
│       ├── tickets.json             # 40 historical resolved support tickets across 8 categories
│       └── new_tickets_example.json # 4 test incoming tickets (NT-1 to NT-4)
└── tests/
    └── test_assistant.py            # 8 unit & integration tests verifying all 4 test cases & edge cases
```

---

## How the 4 Example Tickets (`new_tickets_example.json`) Are Handled

| Incoming Ticket | TRV Tier & Top Score | Top Match & Behavior |
| :--- | :--- | :--- |
| **`NT-1`** (*Can't log in, password reset email never arrives*) | **🟡 Tier 2: Partial Match** (`72%`) | Matches **`T-1001`** on feature (`password-reset`), but flags **Symptom Divergence** (in `T-1001` the email arrived and the password was rejected; in `NT-1` the email never arrived). Pairs `T-1001` with **`T-1003`** (*2FA SMS not received — carrier block*) and warns the agent in bullet #1. |
| **`NT-2`** (*Chart shows nothing for the revenue tab*) | **🟢 Tier 1: Direct Match** (`98%`) | Matches **`T-1201`** (*Revenue chart shows blank instead of bars*) with full `WHERE` + `WHAT` alignment, bridging `"shows nothing / white space"` $\leftrightarrow$ `"blank white box"`. |
| **`NT-3`** (*Getting charged twice this month*) | **🟢 Tier 1: Direct Match** (`98%`) | Matches **`T-1303`** (*Charged twice for the same billing cycle*) with full alignment and recommends refunding the retry duplicate + verifying idempotency keys. |
| **`NT-4`** (*Our data warehouse integration with Snowflake is failing*) | **🟠 Tier 3: Novel Issue Abstention** (`49%`) | Detects `snowflake` as an **unseen third-party entity** absent from `tickets.json`. Abstains from claiming an exact match, classifies into `sync`, and provides a 3-step integration triage checklist from **`T-1105`** (HubSpot OAuth) and **`T-1101`** (Salesforce token). |

---

## Known Limitations & What Is Not Handled Yet (`v1` Scope)

1. **In-Memory Index Storage:** `v1` loads `tickets.json` into memory on startup and stores newly promoted resolutions in process memory. In production (`v2`), promoted resolutions and embeddings would persist to PostgreSQL (`pgvector` + FTS).
2. **Rule & Subword Semantic Layer vs. Neural Embeddings:** To guarantee zero-config offline execution without external API keys or multi-gigabyte model downloads, `v1` uses curated SaaS support concept patterns + character 4-grams alongside weighted BM25. Unseen slang phrasing outside the concept vocabulary relies on subword/stem overlap until paired with dense embeddings in `v2`.
3. **No Temporal Decay or Version Gating Yet:** All resolved tickets are weighted equally regardless of `created_at` age. In `v2`, older tickets will decay in rank or require re-verification after major product releases.
4. **No Automated PII Redaction or CRM Auth:** `v1` assumes pre-sanitized ticket text and does not implement OAuth/RBAC or PII scrubbing.
