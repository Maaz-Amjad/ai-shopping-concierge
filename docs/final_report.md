# Final Report: AI-Powered Semantic Product Discovery, Personalized Recommendation & Conversational Shopping Concierge

Maaz Amjad | Atomcamp AI Bootcamp Final Project (NLP/LLM track) | October 2026

> **Status (29 Oct, final):** results numbers come from `eval/results/` (API call counts from the call log). Every
> product in every reported top 10, including the personalized system's, has a relevance label.

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
| G2 semantic retrieval | dense vs TF-IDF on NDCG@10, with CIs | **0.65 vs 0.40**; paired difference +0.25 [0.05, 0.48] |
| G3 constraint enforcement | adherence ≥ 95 % | **100 %** for the filtered systems (vs 43–63 % without filters; filtered systems are scored on 8 queries because the 2 no-result queries return nothing) |
| G4 personalization | no material relevance loss; beat popularity on next-item HitRate@10 | next item: **6.3 % vs 2.8 %** (CIs do not overlap); relevance: small loss, **−0.026** NDCG@10 [−0.044, −0.007]; precision@10 unchanged (−0.001 [−0.015, 0.010]) |
| G5 grounded answers | 0 IDs outside the retrieved set; mismatches reported | **0** outside IDs; 22/22 answers grounded (21 on the first try); 0 price or brand mismatches in accepted answers |
| NFR-1 latency | retrieval p95 < 500 ms; end-to-end p95 < 6 s | retrieval **11 ms**; generation alone p95 **12.7 s** (target not met) |

Three results are neutral or negative and are reported as such: filter-first search (the proposed refinement) returns
exactly the same top-10 as the proposal's post-filter design on every test query; personalization, while it predicts the
next purchase better than popularity, slightly lowers ranking quality for stated queries (NDCG@10 −0.026); and
end-to-end latency misses its target because of the free-tier LLM endpoint.

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
33 successful Gemini 3.8 Flash calls (plus 42 failed attempts, mostly 503 "overloaded"; counts from the call log in
`artifacts/llm_calls.jsonl`, as of 28 Oct).

## 5. Evaluation design

- **Systems:** S1 TF-IDF, S2 BM25, S3 dense, S4 dense + post-filter, S5 dense + filter-first, S5x = S5 + exclusion terms
  ("not stainless"), S6 = S5 + personalization.
- **Relevance labels:** the union of every system's top 10 per query (659 pairs) was graded 0/1/2 by Gemini 3.8 Flash
  (a different model from the generator). A blind, stratified 25 % sample of those (167 pairs) was re-graded by hand;
  70 further pairs that only S6 surfaces were judged later and are judge-only (so 23 % of all 729 labels are human-checked); the judge
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
| S1 TF-IDF | 0.46 | 0.40 [0.24, 0.58] | 0.43 |
| S2 BM25 | 0.51 | 0.51 [0.33, 0.70] | 0.44 |
| S3 dense | 0.67 | 0.65 [0.45, 0.82] | 0.63 |
| S4 post-filter | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5 filter-first | 0.79 | 0.74 [0.57, 0.89] | 1.00 |
| S5x + exclusions | 0.81 | 0.75 [0.59, 0.90] | 1.00 |
| S6 personalized | 0.79 | 0.71 [0.55, 0.87] | 1.00 |

Paired differences in NDCG@10: S1 → S3 **+0.25 [0.05, 0.48]**; S3 → S5 +0.09 [−0.14, 0.34]; S4 → S5 **0.00**;
S5 → S5x +0.01 [0.00, 0.04]; S5 → S6 −0.02 [−0.04, −0.01] (section 6.3). With 15 queries these are indicative, not significance claims.

- **Semantic retrieval** gives the clearest gain, and the only interval that excludes zero.
- **Filters** guarantee the limits (adherence 1.00), and raise relevance on average, but the interval spans zero:
  filters only act on queries with stated limits, and one wrong filter (q09, section 7) hurt.
- **Filter-first vs post-filter is a neutral result.** Because S4 keeps widening K until 10 products survive, it found the
  same top 10 as S5 on all 15 test queries. The difference is cost and pool size: S4 needed K = 800–1,000 on 3 queries,
  and its candidate pool was smaller on 11 of 15 queries (median 60 vs 200), which limits re-ranking.

A post-hoc view without the two no-result queries (whose ranking metrics reward returning *something*) gives NDCG@10 of
0.43 (S1), 0.63 (S3) and 0.79 (S5); it is secondary because it was chosen after seeing the results.

### 6.2 Query understanding

Dev F1 went from 0.80 (first prompt) to 1.00 (final prompt) through tuning on dev; on test the final prompt scores
**0.94**. It never missed a stated limit. Its two errors were added categories the user did not state.

### 6.3 Personalization

- **Experiment B (next item, 3,995 users):** HitRate@10 random 0.15 %, popularity 2.78 % [2.28, 3.28], profile 4.01 %
  [3.40, 4.66], **blend 6.31 % [5.61, 7.03]**; NDCG@10 shows the same order. Popularity is counted from earlier
  interactions only (an earlier version counted the held-out purchase too; the verifier caught it and it was fixed).
- **Experiment A (query relevance, 8 users with 10+ interactions, test queries):** personalization changes the
  results (Kendall tau 0.80 over the pool, 80 % of the top 10 kept) and costs a little relevance: NDCG@10 drops by
  **−0.026 [−0.044, −0.007]** (averaged over users, then queries; −0.024 [−0.042, −0.007] in the paired table), and all 8
  users are slightly negative (−0.014 to −0.038). Of 104 (user, query) pairs, 48 lose, 43 are unchanged and 13 gain;
  the largest losses are q10 and q19 (−0.08) and q02 (−0.06). Precision@10 barely moves (−0.001 [−0.015, 0.010]), so S6
  keeps the relevant products but ranks some of them lower. The spec expected a small or neutral effect: it is small
  and negative. On dev the same weights gave +0.02, so 10 dev queries were a weak guide here. Combined with experiment B, personalization helps predict what a user buys next but does not help a user
  who has stated what they want.
- **Weight sensitivity (dev, judged items only):** across w_pers 0–0.5 and w_pop 0–0.1 the change in NDCG@10 stays between
  0.00 and +0.04 (+0.019 at the spec weights); what grows with w_pers is the rank shift (Kendall tau 0.76 → 0.43).

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

Failures were tagged by deterministic rules (the spec taxonomy, with *catalog gap* widened to "fewer than 10 products
satisfy the parsed limits" besides missing prices; see section 9) and checked by reading the examples:

- **Parse error → over-filtering (q09, "Frigidaire brand ice maker assembly").** The parser added the category
  *Ice Makers*, but the wanted parts are filed under *Refrigerator Parts & Accessories*; the filter removed them and
  NDCG@10 fell from 0.95 (S3, no filters) to 0.22. This is the main risk of LLM-parsed filters: a plausible but unstated
  limit silently removes the right products.
- **Harmless parse error (q15).** The same kind of added category, but it matched the products the user wanted.
- **Borderline semantic retrieval failure (q17, "replacement wick filter for my bedroom humidifier").** Dense search
  reaches NDCG@10 0.48, just under the 0.5 failure threshold (0.503 before the S6 items joined the pool), so this tag
  depends on the threshold and should be read as borderline.
- **Catalog gap (q24, q25).** No Bosch countertop dishwasher and no LG freezer under $50 exist; the system correctly
  returns nothing and suggests a limit to relax.
- **Label conflicts.** On q24, five full-size Bosch dishwashers were graded 1 by the judge and by hand although they
  break the stated *countertop* limit, so S5 scores 0 for correctly returning nothing. Nine such conflicts exist across
  the test queries; they are reported rather than relabelled.
- **Personalization drift:** none. No query has S6 more than 0.1 below S5; the largest averaged loss is q10 (−0.08).
  Before the S6 items were labelled, this rule tagged five queries, all because unlabelled items counted as 0.
- **Grounding failures:** none on the test queries (the one grounding rejection was a dev query, fixed by
  regeneration).

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
compatibility wording and user-named brands; the error-analysis tag *catalog gap* also covers "fewer than 10 products
satisfy the parsed limits" (no missing-description check, because cleaning already drops near-empty listings); rendered titles shortened to 10
words; Streamlit and MCP wrapper first cut (pre-agreed cut list), then built after the evaluation was complete, together
with a store connector, a chat assistant and other LLM providers (section 11, approved additions); a fourth,
rating term in the production ranking only (section 12).

## 10. Reproducibility

Seed 42, pinned library versions (`requirements.txt`), temperature 0, pinned model names, and every LLM response cached
by (model, prompt version, schema, settings, prompt). All reported tables are regenerated without LLM calls by
`python -m src.evaluate report --split test`, `python -m src.personalize` and the notebooks `03_pipeline` and
`04_evaluation` (which run with an LLM call budget of zero). 290 unit and integration tests run offline with a fake client.

## 11. Prototype and integrations

The pipeline is packaged three ways, all read-only and built on the same code that was evaluated (with the "not X"
exclusions on, i.e. S5x, plus personalization for returning shoppers): a Streamlit demo
(query box, new or returning shopper, enforced limits, ranked products with score components, similar products,
grounded answer); a store connector (an importer for Shopify, WooCommerce or any CSV/JSON product export, a REST API,
and a two-line drop-in search widget); and an MCP server exposing `search_products` and `get_similar` to AI shopping
agents. Imported stores have no interaction history, so their shoppers are cold-start; a store with no sales data gets
a popularity score of 0 (testing the connector exposed a division by zero there, fixed with a failing test first). The
benchmark results above are for the Appliances catalog only; other stores were tested for function, not relevance.

**Chat assistant.** A conversational layer (`src/agent.py`) sits on top of the same pipeline, in the demo app and as a
website chat bubble. Each turn, the LLM chooses one of eight actions by function calling (search or refine, compare,
similar, save, remove, show shortlist, buy link, ask a question) and Python validates the arguments and runs the
action, so the rule "the LLM extracts, Python enforces" still holds: limits go through the same catalog validation and
hard filters, search answers and clarifying replies go through the grounding validator (compare, save, shortlist and
buy replies are templates filled from catalog fields), and the conversation state the client sends back is
re-validated each turn, including its limits. When the LLM call cap is reached or the model returns an unusable
call, the turn falls back to a plain search answered from catalog fields. The assistant never places orders; "buy"
returns the product page. It was evaluated on scripted conversations, each turn checked for the expected action, the
gold limits (price, brand, "not X"), refinements and grounding. *Refinements* adds up three checks (results got
cheaper, a dropped price limit is gone, a dropped brand is gone); *Grounded* is meaningful for search and clarifying
turns, the others are templates and always count as grounded:

| Run | Turns | Right action | Gold limits | Refinements | Grounded |
|---|---|---|---|---|---|
| v1, 10 conversations | 29 | 29/29 | 9/10 | 3/3 | 28/29 |
| v2, same 10 conversations | 29 | 29/29 | 10/10 | 3/3 | 29/29 |
| v2, 4 held-out conversations | 12 | 12/12 | 7/7 | 3/3 | 12/12 |

v1 exposed two faults. Asked "under $400" after "not stainless steel", the LLM restated the limits and dropped the
exclusion; v2 has the LLM return only what changed (plus explicit "remove" and "new search" signals) and merges the
earlier limits in Python. The validator also flagged a greeting because "I" and "part" are brand names in the catalog;
one-letter brands and all-lowercase brands written in lowercase are no longer counted as mentions (the generation
benchmark was re-run and is unchanged). Both fixes were made test-first. The held-out conversations were written and
committed before the fix and were not used to tune it. With 41 turns these results show the design works; they are not
a precise success rate.

**Other LLM providers.** `src/llm.py` has adapters for Gemini (default), any OpenAI-compatible endpoint (ChatGPT, or
open-weight models served by Ollama, vLLM or LM Studio) and Anthropic's Claude, selected in `config.yaml`. Caching, call
caps, retries, structured output and function calling behave the same for every provider. The adapters are tested
offline only; all reported results are Gemini results, and other models may parse and answer differently. One
difference is known: the Claude adapter sends no temperature setting, so fresh calls are not strictly deterministic and repeatability comes from the response cache.

## 12. Production improvements

After the evaluation, the system was extended toward use in a live store, one change at a time, each tuned on dev
queries only and reported once on test.

### 12.1 Ratings as a ranking signal

Star ratings were not used before: the interaction file supplied only who bought what. The product metadata already
held each product's average rating and number of ratings (all 10,000 products; median 4.4 stars, middle half
4.2–4.6), so ratings became a fourth score term, fixed before any result was seen:

- **S_rat** is the Bayesian average (20 × catalog mean + n × average) / (20 + n), min-max normalized within the
  candidate pool. A product with few ratings stays near the catalog mean; one with none gets the mean. When a
  catalog has no ratings at all (e.g. an imported store), the rating share goes back to the semantic term, so it ranks
  exactly as with the spec weights (a review found it did not at first; fixed with a failing test first).
- **Weight** chosen from {0, 0.05, 0.10, 0.15, 0.20}, taken from the semantic share, by best dev judged-only NDCG@10.
  The rule did not say which shoppers to optimize for, and the reading was settled after seeing dev results. New
  shoppers alone, and the two groups weighted equally, pick 0.10 (new shoppers +0.007 [−0.002, +0.019]); returning
  shoppers alone, and all (shopper, query) rows pooled, pick 0.20. Every difference among weights is under 0.01 and
  inside its interval, so the choice rests on nothing firmer than this: 0.10 was taken because new shoppers are
  every imported store's shoppers. Chosen weights: 0.60 semantic, 0.25 personal, 0.05 popularity, 0.10 rating.

| Test, rating weight 0.10 vs 0 | Judged-only NDCG@10 change [95% CI] | Unjudged in top 10 |
|---|---|---|
| New shoppers (13 queries) | −0.009 [−0.031, +0.009] | 4.6 % |
| Returning shoppers (13 queries, 8 users) | −0.006 [−0.015, +0.003] | 3.1 % |

Both rows compare the same ranking with and without the rating term (the pre-registered "S5 vs S5 + rating"
became this cleaner pairing, since the cold-start ranking already differs slightly from S5 through popularity). Counting
unjudged items as not relevant instead gives larger drops (new −0.026, returning −0.022 vs weight 0), which mostly
reflect the 3–5 % of top-10 items nobody has graded; and at weight 0.05 the returning-shopper interval lies just below
zero (−0.007 [−0.016, −0.000]).

The effect on relevance at 0.10 is **neutral**: both intervals include zero. This is the expected outcome rather than a
failure, because the relevance labels grade fit to the request, not product quality; a rating term moves better-rated
products up among similarly relevant ones, which this benchmark cannot reward. It can only show that relevance is not
materially hurt. For example, for "quiet countertop ice maker under $200" the top three are unchanged and a 3.8-star
model (84 ratings) and a 4.0-star model leave the top five for 4.3- and 4.1-star models. Whether that helps shoppers
would need an online test (click-through or conversion), which this project cannot run.

The rating term is used by the demo app, API, chat assistant and MCP server (`app.ranking` in `config.yaml`); every
experiment above keeps the spec's three-term weights, so all earlier tables reproduce exactly (checked: rebuilt
catalog values and benchmark runs identical). Ratings are shown on product cards and in the chat comparison table, and
store imports read optional rating columns (values outside 1–5 are ignored). Review text is still not used.

### 12.2 Stock and catalog updates

A live store must never recommend what it cannot sell. Imported products now carry a stock flag (in stock, out of
stock, or unknown), read from what each platform exports: Shopify inventory quantity, "continue selling when out of
stock" and untracked inventory (combined over a product's variant rows), the availability of each variant from a
Shopify store link, WooCommerce "In stock?" and backorders (a variable product is in stock when any variation is), or
any availability column. Out-of-stock products are removed in the same deterministic mask as the other hard limits,
so search, similar products and the chat never suggest them; unknown stock counts as available. A product saved to the
shortlist that later sells out is labelled, "buy" says so, and when every match is sold out the no-result answer says
how many are out of stock.

Stores are kept current by re-reading their source (`--refresh`, optionally on a schedule, or a button in the app).
Only products whose text changed are embedded again; the new files are swapped in together (the previous ones are
restored if a swap fails) and the API reloads the store. The unit tests cover each platform's stock fields, the
incremental update and a failed save. Two one-off manual checks (not saved as result files): an unchanged 300-product
store updated in about 3 seconds without loading the embedding model, and on a copy of the sample export with 20
products marked sold out, none appeared in five searches or in the similar-product lists of 30 products. The benchmark
catalog has no stock data, so no result above changes.

### 12.3 Adaptive personalization (negative on dev, kept off)

Section 6.3 found that personalization costs some relevance when the shopper states what they want. The obvious fix
is to give the shopper's history less weight when the request states a hard limit (price, category or brand),
moving that share to the semantic term. "States a limit" means the parser extracted one, as it would at run time,
so parser mistakes count too (test q15, a descriptive query, counts as limited through an added category; dev q12, a
category query, does not). The rule and the grid were fixed before any result: keep a share g of the
personal weight, g from {1, 0.5, 0.25, 0}, chosen on dev by judged-only NDCG@10 over the 8 experiment-A shoppers,
ties to the larger g.

| g (share of w_pers kept with limits) | Dev: change vs g = 1 [95% CI] | Test: change vs g = 1 [95% CI] |
|---|---|---|
| 0.5 | −0.002 [−0.018, +0.013] | +0.009 [+0.0005, +0.021] |
| 0.25 | −0.002 [−0.022, +0.016] | +0.016 [−0.002, +0.039] |
| 0 | −0.005 [−0.029, +0.016] | +0.017 [−0.004, +0.045] |

Dev chose **g = 1, i.e. no change**: on dev, personalization helped the queries with limits (+0.012 vs S5). Test,
run once afterwards, points the other way: shrinking the personal weight would have recovered most of the
personalization loss on test (S6 vs S5 from −0.026 to −0.009 at g = 0). The two splits disagree, and dev holds only
four judged queries with limits, so neither is decisive. Switching to g = 0.5 now would be choosing on test, so the
feature ships switched off (`pers_factor_with_limits: 1.0` in `app.ranking`), ready to turn on once a larger dev set
supports it. Reproduce with `python -m src.evaluate adaptive --split dev|test`.

### 12.4 Running for a real store: API hardening and webhooks

The API gained what a public deployment needs, without changing any result: a per-client rate limit (30 requests a
minute for the LLM endpoints, 120 for the rest; 429 with Retry-After), optional per-store API keys, a request log with
no query text or IP address (NFR-5), and a Dockerfile (written, not built here: Docker is not installed on the
development machine). Stores can now push changes: Shopify and WooCommerce product webhooks are checked with the
platform's HMAC-SHA256 signature, repeated deliveries are ignored, and each change updates or removes just that product,
re-embedding only it. Checked live against the running server on a copy of the 300-product store: a signed update was
applied in under a second (price and "out of stock" stored), its repeat was ignored, a forged signature was refused
(401), and the 121st request in a minute was refused (limit 120). A review then hardened it further: webhook bodies are
capped at 2 MB, unknown stores answer like a bad signature, API keys are checked before a store is loaded, the client
address behind a proxy is the one the proxy added, a renamed product replaces its old entry, fields a webhook does not
carry (brand, ratings, a CSV store's category) keep their stored values, a Shopify delete for a store imported from a
CSV is resolved against the live store when its address is known, and a lock file keeps the API, the scheduled refresh
and the app from writing a store at the same time. Each product also keeps the platform's last-changed time (inside
the signed body): an older update or "unpublish" is skipped, deleted products are remembered for 30 days so a late
update cannot bring them back, and a refresh (possibly from a cached snapshot) never replaces a product a newer webhook
already updated; WooCommerce delete messages carry no time, so a delete arriving after a restore still removes the
product until the next refresh. Webhooks are paired with the scheduled refresh (daily is enough),
because a missed or out-of-order delivery is only corrected by the next full read. A key placed in a public web page
is visible to anyone, so for the browser widget the rate limit and the origin allow-list remain the real protection.

### 12.5 Hybrid keyword and semantic retrieval

Appliance shoppers often type part or model numbers, where keyword matching beats embeddings (BM25 reached 0.51
NDCG@10 against TF-IDF's 0.40 in section 6.1). Inside the same hard-filter mask, retrieval now blends the two:
(1 − a) × dense cosine + a × BM25, each min-max normalized over the allowed products, so a = 0 is exactly S5. The grid
{0, 0.1, 0.2, 0.3, 0.5} and the rule (best dev judged-only NDCG@10, ties to the smaller a) were fixed first; dev chose
**a = 0.5**.

| a = 0.5 vs S5 (no personalization) | Judged-only NDCG@10 change [95% CI] | Plain NDCG@10 change | Unjudged in top 10 | Part-number query |
|---|---|---|---|---|
| Dev (10 queries, used to choose) | +0.050 [−0.096, +0.267] | −0.085 | 22 % | q21: 0.94 → 0.96 |
| Test (15 queries, run once) | +0.044 [−0.010, +0.109] | −0.025 | 12 % | q22: 0.60 → 1.00 |

On products that have relevance labels, hybrid retrieval ranks better on both splits, and the part-number query on
test becomes perfect; the interval still includes zero, so the gain is indicative. Counting unlabelled products as not
relevant gives a small loss instead, because keyword matching brings in products no earlier system retrieved (12 % of
the test top 10). The selection leans on this: the unjudged share grows with a, the judged-only reading (which closes
the gaps they leave) rose with it while the plain reading fell, so the rule tends to pick the largest a. Neither
reading is guaranteed to be right; judging the new products with the LLM judge settles it. The "cosine" shown per
product stays the true cosine; the blended score only sets the order. The
chosen a sits at the edge of the grid, so a larger value was not tried. BM25 adds about 12 ms per query (p95 29 ms) and
1.3 s to load a 10,000-product store. The app, API, chat and MCP server use a = 0.5 (`app.retrieval`); every
benchmark table above keeps pure dense retrieval. Reproduce with `python -m src.evaluate hybrid --split dev|test`.

### 12.6 Faster replies: products first

Generation dominates the wait (p95 12.7 s, section 6.5) while retrieval and ranking take milliseconds, so the new
`/search/stream` and `/chat/stream` endpoints send the ranked products as soon as they exist and the written answer
when it is ready; the website widget shows the product cards first with "writing a short summary…". The answer is not
streamed word by word on purpose: it must pass the grounding validator before anyone sees it, so unchecked text is
never shown. This shortens the time until a shopper sees products to roughly the query-understanding call plus
retrieval; the time until the full answer is unchanged. Not measured live (that needs fresh LLM calls); the order of
events is covered by tests.

### 12.7 Categories the shopper did not state (post-hoc)

The error analysis (section 7) found the parser adding a category the shopper never said, which over-filters. Both
such cases are test queries (q09, q15) and dev has none, so nothing could be tuned on dev and this change is
**post-hoc**: it was motivated by test errors, so its test result is not an unbiased estimate. The rule is the
specification's own ("extract only constraints the user states", spec 7.2), not a patch for one query: a new request
keeps a category only if it names it (the category's words, singular or plural). Limits carried over in the chat are
not re-checked, so "only GE" after "fridge water filters" keeps the category. On the saved parses (no LLM calls) it
changes nothing on dev; on test it removes the invented category in q15 (parse F1 0.938 → 0.968, NDCG@10 and adherence
unchanged) and does not fix q09, where "ice maker" is stated but names the part's appliance, not the product. That one
remains a known limitation. Reproduce with `python -m src.evaluate parse-rule --split dev|test`. The tables in section
6 keep the original parses.

### 12.8 Completing the labels for 12.1 and 12.5

Both earlier results left products unlabelled. The same LLM judge then graded the 33 test (query, product) pairs that
the two chosen settings put in their top 10 and nobody had graded (32 done; one pair is still open after the free
quota ran out; grades: 19 relevant, 11 partly relevant, 2 not relevant). These labels are kept in a separate file
(`judge_labels_extra.csv`) and were not checked by a person, unlike the 23 % sample of the main labels; every table in
section 6 keeps the original labels. Adding relevant products also raises the best achievable score of their queries,
so the S5 baseline moves too (NDCG@10 0.738 → 0.730); each comparison below uses the same labels on both sides.

| Test, with the completed labels | NDCG@10 change [95% CI] | Unjudged in top 10 |
|---|---|---|
| Hybrid a = 0.5 vs S5 (12.5) | **+0.037** [−0.017, +0.104] on judged products; +0.033 counting the rest as not relevant | 0.8 % |
| Rating weight 0.10 vs 0, new shoppers (12.1) | −0.009 [−0.032, +0.008] | 0 % |
| Rating weight 0.10 vs 0, returning shoppers (12.1) | −0.007 [−0.015, +0.000] | 0 % |

The two readings of the hybrid result now agree: keyword matching brought in mostly relevant products, and the gain is
modest and indicative (the interval includes zero). The rating result stays neutral. Reproduce with
`python -m src.evaluate rating|hybrid --split test --completed`.

## 13. Future work

Feed the error analysis back into the parser (penalize categories the user did not state); evaluate on a larger,
multi-annotator benchmark; reduce generation latency (paid tier, shorter output, streaming); move masked search to a
vector database with metadata filtering for larger catalogs; test personalization online; and give the personal score a smaller weight, or apply it only to
break near-ties, when the query states clear intent.
