"""
Stage 1: Contraction-Safe Preprocessing, Semantic Concept Mapping & Diagnostic Facet Extraction.

Decomposes support tickets into orthogonal diagnostic facets:
- WHERE (Product Surface / Feature / Third-Party Entity)
- WHAT  (Failure Mode / Observable Symptom)
- DOMAIN (Support Category Signals)
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# Contraction normalization prevents splitting "can't" -> ["can", "t"] (a bug in starter ticket_utils.py)
CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "don't": "do not",
}

# Stopwords including support-ticket conversational filler that caused false positives in starter_kit
STOPWORDS: Set[str] = {
    "a", "about", "above", "across", "after", "again", "against", "all", "almost",
    "also", "although", "always", "am", "among", "an", "and", "another", "any",
    "anyone", "anything", "are", "around", "as", "at", "back", "be", "became",
    "because", "become", "becomes", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "customer", "customers", "day",
    "days", "did", "do", "does", "doing", "done", "down", "during", "each", "either",
    "else", "even", "ever", "every", "few", "for", "from", "further", "get", "gets",
    "getting", "give", "given", "go", "going", "gone", "got", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "however", "hour",
    "hours", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "keep",
    "keeps", "know", "last", "less", "let", "like", "likely", "made", "make",
    "makes", "many", "may", "me", "might", "more", "most", "much", "must", "my",
    "near", "need", "needs", "needed", "new", "newer", "next", "no", "nor", "not",
    "now", "notice", "noticed", "notices", "of", "off", "often", "on", "once",
    "one", "only", "onto", "or", "other", "others", "ought", "our", "ours", "out",
    "over", "own", "past", "per", "perhaps", "please", "put", "quite", "rather",
    "really", "recent", "recently", "report", "reported", "reports", "said", "same",
    "saw", "say", "says", "see", "seem", "seemed", "seems", "seen", "several",
    "shall", "she", "should", "show", "showed", "showing", "shown", "shows", "since",
    "so", "some", "somebody", "someone", "something", "sometimes", "still", "such",
    "take", "taken", "takes", "taking", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "therefore", "these", "they", "thing",
    "things", "think", "this", "those", "though", "through", "thus", "ticket",
    "tickets", "time", "times", "to", "today", "together", "too", "took", "toward",
    "tried", "tries", "try", "trying", "turn", "turned", "up", "upon", "us", "use",
    "used", "uses", "using", "very", "via", "want", "wanted", "wants", "was", "way",
    "we", "well", "went", "were", "what", "whatever", "when", "where", "whether",
    "which", "while", "who", "whom", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "able", "unable", "issue",
    "issues", "problem", "problems", "work", "working", "works", "worked", "fine",
    "main", "page", "app", "application", "system", "account", "accounts",
}

# Semantic concept clusters bridge customer vocabulary and internal engineering resolution terms
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "concept_blank_empty": [
        r"\bblank\b", r"\bempty\b", r"shows?\s+nothing", r"white\s+space",
        r"white\s+box", r"no\s+bars", r"nothing\s+for", r"empty\s+dashboard",
    ],
    "concept_duplicate_charge": [
        r"charged\s+twice", r"two\s+charges", r"two\s+identical\s+charges",
        r"duplicate[\s\-]+charge", r"double[\s\-]+charg", r"billed\s+twice",
    ],
    "concept_delivery_missing": [
        r"never\s+arrives?", r"not\s+being\s+received", r"has\s+not\s+shown\s+up",
        r"hasn't\s+shown\s+up", r"stopped\s+arriving", r"haven't\s+gotten",
        r"not\s+arriving", r"never\s+received", r"checking\s+spam",
    ],
    "concept_password_reset": [
        r"password\s+reset", r"reset\s+their\s+password", r"reset\s+email",
        r"temporary\s+password", r"reset\s+link",
    ],
    "concept_auth_rejected": [
        r"rejected\s+as\s+incorrect", r"wrong\s+password", r"failed\s+login",
        r"locked\s+out", r"account\s+locked", r"authentication\s+error",
    ],
    "concept_timeout_slow": [
        r"times?\s+out", r"\btimeout\b", r"30\+?\s*seconds", r"10[\-\s]15\s+seconds",
        r"extremely\s+slow", r"spinning\s+indefinitely", r"loading\s+spinner",
        r"sluggish", r"unresponsive", r"stuck\s+at", r"in\s+progress",
    ],
    "concept_data_gap_timezone": [
        r"missing\s+recent", r"missing\s+last\s+week", r"missing\s+the\s+most\s+recent",
        r"last\s+2\s+days.*missing", r"\btimezone\b", r"\butc\b", r"data[\s\-]+gap",
    ],
    "concept_data_discrepancy": [
        r"don't\s+add\s+up", r"do\s+not\s+add\s+up", r"97%", r"100%",
        r"don't\s+match", r"do\s+not\s+match", r"noticeably\s+different",
        r"incorrect\s+seat\s+count",
    ],
    "concept_permission_access": [
        r"greyed\s+out", r"\bdisabled\b", r"read[\s\-]+only", r"\bviewer\b",
        r"revoke\s+access", r"former\s+employee", r"deactivate", r"custom\s+role",
        r"default\s+role", r"admin\s+access", r"can't\s+see\s+any\s+dashboards",
    ],
    "concept_third_party_sync": [
        r"\bintegration\b", r"\bsync\b", r"\bsyncing\b", r"\bwarehouse\b",
        r"connection\s+error", r"\boauth\b", r"api\s+token",
    ],
}

# Diagnostic Facet Definitions: WHERE (Product Surface / Feature)
WHERE_FACET_RULES: Dict[str, List[str]] = {
    "password-reset": [r"password\s+reset", r"reset.*password", r"temporary\s+password"],
    "sso-login": [r"\bsso\b", r"\bsaml\b", r"\bokta\b"],
    "2fa-sms": [r"\b2fa\b", r"two[\s\-]+factor", r"\bsms\b", r"verification\s+code"],
    "login-session": [r"\bsession\b", r"logged\s+out", r"account\s+locked", r"login\s+attempts"],
    "revenue-chart": [r"revenue\s+chart", r"revenue\s+tab", r"revenue\s+dashboard", r"\bbars\b"],
    "pie-chart": [r"pie\s+chart", r"percentages?", r"segments?"],
    "dashboard-ui": [r"\bwidgets?\b", r"smaller\s+screens?", r"responsive", r"loading\s+spinner"],
    "date-filter": [r"date\s+range", r"date\s+filter", r"custom\s+date"],
    "subscription-billing": [
        r"billing\s+cycle", r"monthly\s+subscription", r"monthly\s+charge",
        r"card\s+statement", r"plan\s+downgrade", r"annual\s+plan", r"annual\s+billing",
    ],
    "invoice-seats": [r"\binvoice\b", r"seat\s+count", r"\bseats\b"],
    "payment-method": [r"credit\s+card", r"payment\s+method", r"postal\s+code", r"zip\s+code"],
    "csv-export": [r"csv\s+export", r"exported\s+csv", r"\bcsv\b"],
    "pdf-export": [r"pdf\s+report", r"\bpdf\b"],
    "scheduled-report": [r"scheduled.*report", r"weekly.*report", r"report\s+generation\s+queue"],
    "export-controls": [r"export\s+button", r"download\s+reports"],
    "role-permissions": [
        r"viewer\s+role", r"viewer\s+permissions", r"custom\s+role",
        r"default\s+role", r"team\s+member", r"former\s+employee",
    ],
    "data-sync-integration": [
        r"\bsalesforce\b", r"\bhubspot\b", r"\bsnowflake\b", r"data\s+warehouse",
        r"nightly.*sync", r"sync\s+job", r"re[\s\-]?sync", r"\bintegration\b",
    ],
    "webhooks-api": [r"\bwebhooks?\b", r"rate[\s\-]+limit", r"\b429\b", r"\b500\b", r"api\s+key"],
    "database-search": [r"search\s+feature", r"2\s+million\s+records", r"large\s+datasets?"],
}

# Diagnostic Facet Definitions: WHAT (Symptom / Failure Mode)
WHAT_FACET_RULES: Dict[str, List[str]] = {
    "delivery-failure": [
        r"never\s+arrives?", r"not\s+being\s+received", r"has\s+not\s+shown\s+up",
        r"hasn't\s+shown\s+up", r"stopped\s+arriving", r"haven't\s+gotten",
        r"silently\s+misses", r"missing\s+from.*payload",
    ],
    "blank-or-empty-ui": [
        r"\bblank\b", r"\bempty\b", r"shows?\s+nothing", r"white\s+space",
        r"white\s+box", r"no\s+bars", r"can't\s+see\s+any\s+dashboards",
    ],
    "duplicate-occurrence": [
        r"charged\s+twice", r"two\s+charges", r"two\s+identical\s+charges",
        r"duplicate\s+records", r"duplicate\s+entries", r"duplicate[\s\-]+charge",
    ],
    "credential-or-connection-rejection": [
        r"rejected\s+as\s+incorrect", r"authentication\s+error", r"connection\s+error",
        r"integration.*failing", r"locked\s+out", r"unable\s+to\s+update\s+credit\s+card",
    ],
    "timeout-or-slowness": [
        r"times?\s+out", r"\btimeout\b", r"30\+?\s*seconds", r"extremely\s+slow",
        r"spinning\s+indefinitely", r"stuck\s+at", r"sluggish", r"unresponsive",
        r"hours\s+late", r"expires\s+too\s+quickly",
    ],
    "data-or-calculation-discrepancy": [
        r"don't\s+add\s+up", r"do\s+not\s+add\s+up", r"don't\s+match",
        r"do\s+not\s+match", r"incorrect\s+seat\s+count", r"not\s+reflected",
        r"discount\s+not\s+applied", r"not\s+updating", r"missing\s+recent",
        r"missing\s+last\s+week", r"hasn't\s+updated",
    ],
    "permission-or-state-violation": [
        r"greyed\s+out", r"\bdisabled\b", r"able\s+to\s+log\s+in",
        r"still\s+works", r"can\s+edit", r"not\s+saving", r"revert\s+after\s+saving",
        r"wrong\s+default\s+role", r"widgets.*overlap",
    ],
    "http-or-rate-error": [
        r"\b429\b", r"rate[\s\-]+limit", r"\b500\s+errors?\b",
    ],
}

# Known external vendor / technology names to detect unseen third-party integrations (like Snowflake)
KNOWN_TECH_ENTITIES: Set[str] = {
    "snowflake", "salesforce", "hubspot", "okta", "saml", "bigquery", "redshift",
    "databricks", "looker", "tableau", "powerbi", "stripe", "paypal", "slack",
    "teams", "jira", "zendesk", "intercom", "postgres", "mysql", "mongodb",
    "kafka", "airflow", "dbt", "fivetran", "azure", "aws", "gcp",
}


def stem_token(word: str) -> str:
    """Lightweight deterministic suffix stemmer tailored for support terminology."""
    if len(word) <= 3:
        return word
    # Handle common irregulars / plurals
    irregulars = {
        "charged": "charge",
        "charges": "charge",
        "charging": "charge",
        "duplicated": "duplicate",
        "duplicates": "duplicate",
        "syncing": "sync",
        "synced": "sync",
        "syncs": "sync",
        "exported": "export",
        "exporting": "export",
        "exports": "export",
        "invoices": "invoice",
        "invoiced": "invoice",
        "payments": "payment",
        "permissions": "permission",
        "integrations": "integration",
        "dashboards": "dashboard",
        "widgets": "widget",
        "charts": "chart",
        "bars": "bar",
        "filters": "filter",
        "filtering": "filter",
        "filtered": "filter",
        "queries": "query",
        "webhooks": "webhook",
        "sessions": "session",
        "emails": "email",
        "users": "user",
        "roles": "role",
        "records": "record",
        "transactions": "transaction",
        "arrives": "arrive",
        "arrived": "arrive",
        "arriving": "arrive",
        "received": "receive",
        "receiving": "receive",
        "failing": "fail",
        "failed": "fail",
        "fails": "fail",
        "locked": "lock",
        "lockouts": "lockout",
        "lockout": "lock",
        "renders": "render",
        "rendering": "render",
        "rendered": "render",
    }
    if word in irregulars:
        return irregulars[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4 and not word.endswith(("ses", "xes", "zes")):
        return word[:-1]
    if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def preprocess_raw_text(text: str) -> str:
    """Lowercase and expand contractions before punctuation stripping."""
    lowered = text.lower()
    # Normalize curly apostrophes
    lowered = lowered.replace("’", "'").replace("‘", "'")
    for contraction, expansion in CONTRACTIONS.items():
        lowered = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, lowered)
    # Strip possessive 's cleanly so "customer's" -> "customer"
    lowered = re.sub(r"'s\b", "", lowered)
    return lowered


def normalize_tokens(text: str, include_concepts: bool = True) -> List[str]:
    """
    Tokenize, remove stopwords, stem, and optionally append semantic concept tokens.
    Returns a list of tokens (preserving multiplicity for TF/BM25 scoring).
    """
    cleaned = preprocess_raw_text(text)
    raw_words = re.sub(r"[^a-z0-9\s]", " ", cleaned).split()
    tokens: List[str] = []
    for w in raw_words:
        if w in STOPWORDS or len(w) <= 1:
            continue
        stemmed = stem_token(w)
        if stemmed not in STOPWORDS and len(stemmed) > 1:
            tokens.append(stemmed)

    if include_concepts:
        for concept_name, patterns in CONCEPT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, cleaned):
                    tokens.append(concept_name)
                    break

    return tokens


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Extract character n-grams over content words for subword similarity."""
    tokens = [t for t in normalize_tokens(text, include_concepts=False) if not t.startswith("concept_")]
    joined = " ".join(tokens)
    if len(joined) < n:
        return {joined} if joined else set()
    return {joined[i : i + n] for i in range(len(joined) - n + 1)}


def extract_facets(
    subject: str,
    description: str,
    known_corpus_vocab: Set[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Extract structured diagnostic facets from a ticket:
    - where_facets: Product surfaces / features involved (WHERE)
    - what_facets: Observable failure modes / symptoms (WHAT)
    - concepts: High-level semantic concepts triggered
    - unseen_entities: Named integrations/technologies absent from historical tickets
    """
    combined = preprocess_raw_text(f"{subject} {description}")

    where_matches: List[str] = []
    for facet_name, patterns in WHERE_FACET_RULES.items():
        if any(re.search(p, combined) for p in patterns):
            where_matches.append(facet_name)

    what_matches: List[str] = []
    for facet_name, patterns in WHAT_FACET_RULES.items():
        if any(re.search(p, combined) for p in patterns):
            what_matches.append(facet_name)

    concepts: List[str] = []
    for concept_name, patterns in CONCEPT_PATTERNS.items():
        if any(re.search(p, combined) for p in patterns):
            concepts.append(concept_name)

    # Detect unseen third-party entities or capitalized product names not in historical corpus
    raw_words = set(re.sub(r"[^a-z0-9\s]", " ", combined).split())
    unseen_entities: List[str] = []
    if known_corpus_vocab is not None:
        # Check known tech entities first
        for entity in sorted(raw_words & KNOWN_TECH_ENTITIES):
            if entity not in known_corpus_vocab and stem_token(entity) not in known_corpus_vocab:
                unseen_entities.append(entity)
        # Also check capitalized proper nouns inside the original text (excluding sentence starts)
        orig_text = f"{subject}. {description}"
        for match in re.finditer(r"(?<=[a-z]\s)([A-Z][a-zA-Z0-9]{2,})\b", orig_text):
            proper_word = match.group(1).lower()
            if (
                proper_word not in STOPWORDS
                and proper_word not in known_corpus_vocab
                and stem_token(proper_word) not in known_corpus_vocab
                and proper_word not in unseen_entities
            ):
                unseen_entities.append(proper_word)

    # Fallback readable labels if a custom query doesn't hit predefined rules
    if not where_matches:
        content_toks = normalize_tokens(subject, include_concepts=False)[:2]
        if content_toks:
            where_matches = ["-".join(content_toks)]
    if not what_matches:
        desc_toks = normalize_tokens(description, include_concepts=False)[:2]
        if desc_toks:
            what_matches = ["-".join(desc_toks)]

    return {
        "where_facets": where_matches,
        "what_facets": what_matches,
        "concepts": concepts,
        "unseen_entities": unseen_entities,
    }
