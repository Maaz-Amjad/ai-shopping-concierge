# Intent: Semantic product discovery & shopping concierge
Author: Maaz Amjad (atomcamp AI Bootcamp, NLP/LLM track). Status: accepted. Date: 2026-10-26.

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
