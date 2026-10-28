# Final Report: AI-Powered Semantic Product Discovery, Personalized Recommendation & Conversational Shopping Concierge

Maaz Amjad | Atomcamp AI Bootcamp Final Project (NLP/LLM track) | October 2026

> **Status (28 Oct):** every number below comes from `eval/results/`. Numbers marked *provisional* depend on the
> personalized system S6, 11 % of whose top-10 items are not yet labelled (judge API overloaded); they are scored on
> judged items only and will be refreshed once those items are graded.

## 1. Summary

The project builds and evaluates a shopping assistant over one product catalog (10,000 Amazon appliance listings).
An LLM (Gemini) turns a request such as *"quiet portable countertop ice maker under $150"* into structured limits
plus a descriptive query; Python enforces the limits; BGE embeddings retrieve inside the allowed products; a weighted
score personalizes the order for returning shoppers; and the LLM writes a short answer that a deterministic validator
checks against the catalog.

On 15 held-out test queries (never used for tuning):

| Goal | Target | Result |
|---|---|---|
| G1 query understanding | parse F1 ≥ 0.85 | **0.94** (precision 0.88, recall 1.00) |
| G2 semantic retrieval | dense vs TF-IDF on NDCG@10, with CIs | **0.66 vs 0.41**; paired difference +0.25 [0.05, 0.47] |
| G3 constraint enforcement | adherence ≥ 95 % | **100 %** for the filtered systems (vs 43–63 % without filters) |
| G4 personalization | no material relevance loss; beat popularity on next-item HitRate@10 | next item: **6.3 % vs 2.8 %** (CIs do not overlap); relevance: small loss, **−0.025** NDCG@10 [−0.052, −0.004] on judged items (*provisional*) |
| G5 grounded answers | 0 IDs outside the retrieved set; mismatches reported | **0** outside IDs; 22/22 answers grounded (21 on the first try); 0 price or brand mismatches in accepted answers |
| NFR-1 latency | retrieval p95 < 500 ms; end-to-end p95 < 6 s | retrieval **11 ms**; generation alone p95 **12.7 s** (target not met) |

Two results are neutral or negative and are reported as such: filter-first search (the proposed refinement) returns
exactly the same top-10 as the proposal's post-filter design on every test query, and end-to-end latency misses its
target because of the free-tier LLM endpoint.

## 2. Problem

Keyword search fails on descriptive requests (vocabulary mismatch), similarity search alone does not enforce hard
limits such as a price cap, and neither adapts to the shopper or explains its picks. The design principle throughout is
**"the LLM extracts, Python enforces"**: the LLM never filters, ranks, or decides which products exist.

## 3. Data

- **Catalog:** Amazon Reviews 2023 (McAuley Lab), *Appliances*. From 94,327 raw listings, cleaning removed 9 with no
  title, 12,823 with under 20 characters of description and features, and 4,322 duplicate titles; the 10,000 most
  reviewed of the remaining 77,173 were kept. 24 categories, 2,315 brands; **23.9 % of products have no price**, which
  makes the missing-price policy matter (such products are excluded only when a price filter is active).
- **Interactions:** 24,103 ratings by 3,995 users with at least 5 interactions on catalog products (185 users with 10+).
- **Benchmark:** 25 hand-written queries in 6 strata (price, category/brand, descriptive, negation, exact part number,
  no-result), with gold limits; split 10 dev / 15 test, stratified.

## 4. System

| Stage | Module | Method |
|---|---|---|
| Query understanding | `src/parse.py` | Gemini 3.5 Flash-Lite, temperature 0, JSON schema; category and brand must be catalog values (fuzzy match, else dropped and logged); one retry on invalid output, then semantic-only fallback |
| Constraint resolution | `src/filters.py` | Boolean mask for price, category, brand; null price fails any price limit |
| Retrieval | `src/retrieve.py` | `BAAI/bge-small-en-v1.5` with the BGE query prefix; filter-first exact cosine search over the masked rows (S5); post-filter with K doubling from 100 to 1,000 (S4) |
| Ranking | `src/personalize.py` | `0.7·S_sem + 0.25·S_pers + 0.05·S_pop`, semantic and personal scores min-max normalized per pool; new users get `w_pers = 0` |
| Answer | `src/generate.py` | Structured output (summary + product IDs with reasons) from catalog fields only, wrapped in `<product>` tags as data; validator checks IDs, dollar amounts and brands; one regeneration with the errors, then a template answer |
| No result | `src/filters.py` | Counts how many products remain if each limit is dropped and suggests the best one; no LLM |

Every LLM call goes through one client (`src/llm.py`) that caches responses on disk, rate-limits, retries transient
errors, and logs each call. Over the whole project the pipeline used 61 Flash-Lite calls and the relevance judge
33 successful Gemini 3.8 Flash calls (plus 27 failed attempts, mostly 503 "overloaded").

## 5. Evaluation design

- **Systems:** S1 TF-IDF, S2 BM25, S3 dense, S4 dense + post-filter, S5 dense + filter-first, S5x = S5 + exclusion terms
  ("not stainless"), S6 = S5 + personalization.
- **Relevance labels:** the union of every system's top 10 per query (659 pairs) was graded 0/1/2 by Gemini 3.8 Flash
  (a different model from the generator). A blind, stratified 25 % sample (167 pairs) was re-graded by hand; the judge
  agreed on the exact grade for **81.4 %** and on relevant-vs-not for **91.6 %**, and was the stricter grader. The human
  grade wins wherever both exist.
- **Metrics:** Precision@K, pooled Recall@K, NDCG@K (K = 5, 10), constraint adherence, parse micro-F1, grounding,
  latency. 95 % bootstrap CIs over queries (1,000 resamples, seed 42); paired bootstrap for adjacent systems.
- **Tuning discipline:** the parser prompt (v1 → v2) and the BGE prefix were decided on dev queries only; ranking weights
  were fixed at the spec values and only a sensitivity analysis was run (on dev). Test numbers were computed once the
  systems were frozen.

## 6. Results

### 6.1 Retrieval and filtering (test, 15 queries)

| System | P@10 | NDCG@10 [95 % CI] | Adherence@10 |
|---|---|---|---|
| S1 TF-IDF | 0.46 | 0.41 [0.25, 0.59] | 0.43 |
| S2 BM25 | 0.51 | 0.52 [0.33, 0.71] | 0.44 |
| S3 dense | 0.67 | 0.66 [0.47, 0.83] | 0.63 |
| S4 post-filter | 0.79 | 0.75 [0.60, 0.90] | 1.00 |
| S5 filter-first | 0.79 | 0.75 [0.60, 0.90] | 1.00 |
| S5x + exclusions | 0.81 | 0.77 [0.60, 0.92] | 1.00 |
| S6 personalized (*provisional*) | 0.71 | 0.66 [0.52, 0.81] | 1.00 |

Paired differences in NDCG@10: S1 → S3 **+0.25 [0.05, 0.47]**; S3 → S5 +0.10 [−0.14, 0.35]; S4 → S5 **0.00**;
S5 → S5x +0.01 [0.00, 0.04]; S5 → S6 −0.08 [−0.13, −0.04] counting unlabelled S6 items as irrelevant (−0.025 on
judged items only, section 6.3). With 15 queries these are indicative, not significance claims.

- **Semantic retrieval** gives the clearest gain, and the only interval that excludes zero.
- **Filters** guarantee the limits (adherence 1.00), and raise relevance on average, but the interval spans zero:
  filters only act on queries with stated limits, and one wrong filter (q09, section 7) hurt.
- **Filter-first vs post-filter is a neutral result.** Because S4 keeps widening K until 10 products survive, it found the
  same top 10 as S5 on all 15 test queries. The difference is cost and pool size: S4 needed K = 800–1,000 on 3 queries,
  and its candidate pool was smaller on 11 of 15 queries (median 60 vs 200), which limits re-ranking.

A post-hoc view without the two no-result queries (whose ranking metrics reward returning *something*) gives NDCG@10 of
0.44 (S1), 0.64 (S3) and 0.81 (S5); it is secondary because it was chosen after seeing the results.

### 6.2 Query understanding

Dev F1 went from 0.80 (first prompt) to 1.00 (final prompt) through tuning on dev; on test the final prompt scores
**0.94**. It never missed a stated limit. Its two errors were added categories the user did not state.

### 6.3 Personalization

- **Experiment B (next item, 3,995 users):** HitRate@10 random 0.15 %, popularity 2.78 % [2.28, 3.28], profile 4.01 %
  [3.40, 4.66], **blend 6.31 % [5.61, 7.03]**; NDCG@10 shows the same order. Popularity is counted from earlier
  interactions only (an earlier version counted the held-out purchase too; the verifier caught it and it was fixed).
- **Experiment A (query relevance, 8 users with 10+ interactions, test queries; *provisional*):** personalization
  changes the results (Kendall tau 0.80 over the pool, 80 % of the top 10 kept) and costs a little relevance: on judged
  items NDCG@10 drops by **−0.025 [−0.052, −0.004]**, and all 8 users are slightly negative (−0.010 to −0.043). The median
  (user, query) change is 0; the loss comes from a few queries (q10 −0.14, q19 −0.08, q02 and q09 −0.06). The plain change, which
  counts the 11 % of S6's top 10 that is still unlabelled as irrelevant, is −0.09. The spec expected a small or neutral
  effect: it is small but consistently negative. Dev queries had suggested about +0.01, so 10 dev queries were a weak
  guide here. Combined with experiment B, personalization helps predict what a user buys next but does not help a user
  who has stated what they want.
- **Weight sensitivity (dev, judged items only):** across w_pers 0–0.5 and w_pop 0–0.1 the change in NDCG@10 stays within
  ±0.012, so on labelled items personalization barely affects relevance; what grows with w_pers is the rank shift.

### 6.4 Grounded answers

22 of the 25 queries reach the LLM (3 no-result queries are answered deterministically). 21 answers passed the
validator on the first try and the remaining one after one regeneration; the template fallback was never needed. The
rejected draft cited a product ID that does not exist. No accepted answer contains an ID outside the retrieved set, a
price that is not a listed or user-stated price, or a brand absent from the product data. The LLM-written text stays
within 120 words (max 105); with full product titles, 11 rendered answers exceed 120 words.

Building the validator was itself instructive: its first version flagged a brand the user had named ("not Samsung") and
common-word brands ("Best"), and a later demo showed that matching the catalog spelling exactly missed an invented
"Bosch" (catalog: "BOSCH"). Each case was fixed with a failing test first; the accepted answers did not change.

### 6.5 Latency

Retrieval with filtering: median 4 ms, p95 11 ms; query encoding about 17 ms; ranking under 20 ms. Generation with a live
LLM call: median 6.0 s, p95 12.7 s. Parsing latency was not measured (parses were cached). The 500 ms retrieval target
is met by a wide margin; the 6 s end-to-end target is not.

## 7. Error analysis (test queries)

Failures were tagged by deterministic rules (spec taxonomy) and checked by reading the examples:

- **Parse error → over-filtering (q09, "Frigidaire brand ice maker assembly").** The parser added the category
  *Ice Makers*, but the wanted parts are filed under *Refrigerator Parts & Accessories*; the filter removed them and
  NDCG@10 fell from 0.95 (S3, no filters) to 0.22. This is the main risk of LLM-parsed filters: a plausible but unstated
  limit silently removes the right products.
- **Harmless parse error (q15).** The same kind of added category, but it matched the products the user wanted.
- **Catalog gap (q24, q25).** No Bosch countertop dishwasher and no LG freezer under $50 exist; the system correctly
  returns nothing and suggests a limit to relax.
- **Label conflicts.** On q24, five full-size Bosch dishwashers were graded 1 by the judge and by hand although they
  break the stated *countertop* limit, so S5 scores 0 for correctly returning nothing. Nine such conflicts exist across
  the test queries; they are reported rather than relabelled.
- **Personalization drift (*provisional*).** The rule (S6 NDCG@10 more than 0.1 below S5) tags five queries, but it
  counts unlabelled items as irrelevant; on judged items only, one query crosses it: q10, "beverage refrigerator for cans
  and wine" (−0.14).
- **Semantic retrieval failures and grounding failures:** none on the test queries (the one grounding rejection was a
  dev query, fixed by regeneration).

## 8. Limitations

- **Small benchmark.** 15 test queries give wide intervals; differences are indicative.
- **Pooling bias.** Labels come from the pooled top 10 of S1–S5x, so systems outside the pool (other query prefixes,
  personalization) surface unjudged items; comparisons for those use judged-only lists and report the unjudged share.
  S6's extra items are graded by the same judge in separate per-query batches and are not human-verified.
- **LLM judge.** 81 % exact agreement with a human; the judge is stricter than the human grader.
- **One catalog, English only, offline.** No online A/B test; personalization is evaluated on historic ratings.
- **Validator scope.** Grounding is checked per answer (a brand true for one product could be attached to another),
  only `$` amounts are checked, and any number in the query counts as a stated price.
- **Free-tier API.** 20 judge requests per day (failed requests count) and 503 overloads shaped the schedule; latency
  numbers reflect that endpoint.

## 9. Departures from the specification

All were recorded during the build: 10,000 products (the low end of the spec range); 25 queries (the spec minimum);
local CPU instead of Colab; a separate "price not listed" list for unpriced matches; S5x reported separately from S5;
judged-only metrics for unpooled comparisons; S6 items judged in separate batches; the validator's brand rule extended to
compatibility wording and user-named brands; rendered titles shortened to 10 words; Streamlit and MCP wrapper cut
(pre-agreed cut list).

## 10. Reproducibility

Seed 42, pinned library versions (`requirements.txt`), temperature 0, pinned model names, and every LLM response cached
by (model, prompt version, schema, settings, prompt). All reported tables are regenerated without LLM calls by
`python -m src.evaluate report --split test`, `python -m src.personalize` and the notebooks `03_pipeline` and
`04_evaluation` (which run with an LLM call budget of zero). 146 unit and integration tests run offline with a fake client.

## 11. Future work

Feed the error analysis back into the parser (penalize categories the user did not state); evaluate on a larger,
multi-annotator benchmark; reduce generation latency (paid tier, shorter output, streaming); move masked search to a
vector database with metadata filtering for larger catalogs; and test personalization online.
