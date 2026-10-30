# Intent: Semantic product discovery & shopping concierge
Author: Maaz Amjad (atomcamp AI Bootcamp, NLP/LLM track). Status: accepted 2026-10-26; revised 2026-10-30 (see Revisions).

## Problem
Keyword search misses products described in different words, does not reliably
enforce stated limits (price, category, brand), and rarely reflects a shopper's
history or explains why a product fits.

## Proposed outcome
A reproducible prototype over one Amazon catalog (~10k Appliances products) that
parses a natural-language query into a semantic query plus hard filters (LLM
extracts, Python enforces), retrieves with BGE embeddings filter-first, re-ranks
with a light user-preference signal, and returns 3-5 products with a grounded,
validated explanation. Success is measured on a 25-query benchmark against
TF-IDF, BM25 and dense baselines (S1-S6) with bootstrap CIs; neutral or negative
results are reported as found.

## Affected users and systems
Shoppers (personas P1-P4, simulated), trainer/assessor (Mr. Muhammad Umair),
future store owners. Systems: Amazon Reviews 2023 data, Gemini API (free tier),
local CPU runtime, GitHub repo.

## Constraints
Deadline 28 Oct 2026 23:59 PKT. Gemini free tier only (tight daily quotas),
temperature 0, all LLM calls cached. No personal data sent; keys never committed.
Tune on dev queries only; report on test only. Spec docs are the source of truth.

## Open questions
Report and slide format atomcamp expects? Presentation date? Is the free-tier
judge quota enough, or will billing be enabled?

## Revisions (27–30 Oct 2026)

**Constraints changed.** Deadline moved from 28 to 30 Oct 2026. The relevance judge's free-tier quota (20 requests a
day, failed requests count) was enough with waiting between quota resets; billing was not enabled.

**Outcome achieved.** The benchmark prototype was delivered as intended: parse F1 0.94, dense beats TF-IDF by +0.25
NDCG@10, 100 % constraint adherence, 0 invented product IDs; filter-first vs post-filter and personalization's small
relevance cost were reported as neutral and negative findings ([final report](../docs/final_report.md), sections 1–10).

**Intent widened (approved by the author during the build).** From a research prototype to a store-ready concierge
that a non-technical store owner can plug in: a no-code demo app, REST API and website chat bubble, MCP server, store
connector (Shopify, WooCommerce, any CSV/JSON), a conversational agent that can refine, compare, shortlist and hand
off to the store (never placing orders), any LLM provider, and production extensions (ratings, stock, store updates by
refresh and signed webhooks, hybrid retrieval, faster replies, shopper events, API hardening). The original success
criteria still apply to the benchmark, which keeps the evaluated configuration; each extension was chosen on dev and
reported once on test, including the ones that did not help (final report, sections 11–12;
[requirements addendum](../docs/requirements.md)).

**Affected users and systems, added.** Store owners now use the system directly (app, widget, webhooks, events
summary); shopper events are collected in privacy-limited form (no query text or IP; random browser id renewed every
30 days). Systems added: Shopify and WooCommerce (exports, store links, webhooks), Claude Desktop (MCP), optional
OpenAI-compatible and Anthropic APIs, Docker.

**Open questions, answered.** Judge quota: sufficient with waits, no billing. Report format: Markdown report in the
repository, notebooks and README. Still open: the slide format atomcamp expects and the presentation date.

## Next intent (after submission; to be planned and approved first)

Turn the prototype into a multi-store product: user accounts with one workspace per store; store owners bring their
own LLM and webhook keys (encrypted, with per-store call caps); a dashboard showing how the assistant changes shopping
behaviour and sales (event data, order webhooks, an A/B holdout); a web front end replacing Streamlit; a shared
database and vector index; and free-tier deployment. Success would be measured online (click, save and conversion
rates against a holdout), and offline on a larger, category-diverse benchmark.

