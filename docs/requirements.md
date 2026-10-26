# Requirements Document: AI-Powered Semantic Product Discovery, Personalized Recommendation & Conversational Shopping Concierge

Prepared by Maaz Amjad | Atomcamp AI Bootcamp Final Project (NLP/LLM track) | 5 October 2026

## 1. Document Purpose

This document defines **what** the system must do and how its success is judged. The companion Technical Specification defines **how** it is built. Items marked **\[Refinement\]** go beyond the submitted proposal and may change after trainer feedback.

## 2. Background and Problem Statement

Keyword-based e-commerce search struggles with natural, descriptive queries such as 'breathable running jacket for rainy weather under $80'. Three problems drive this project:

- **Vocabulary mismatch:** relevant products described with different words are missed by keyword matching.
- **Constraint handling:** semantic similarity alone does not reliably enforce hard limits such as price, category, or brand.
- **Limited personalization and explanation:** results rarely reflect a shopper's history or explain why a product fits the request.

The system combines LLM query understanding, dense semantic retrieval, deterministic filtering, lightweight personalization, and grounded LLM response generation in a single prototype for one product catalog.

## 3. Goals and Success Criteria

Targets below are hypotheses to validate, not guarantees. Negative or neutral results are acceptable if they are measured honestly and analyzed.

| ID | Goal | Measurable success criterion |
| --- | --- | --- |
| G1 | Understand natural-language shopping queries | Filter-extraction F1 against hand-annotated gold constraints; target 0.85 or higher |
| G2 | Retrieve semantically relevant products | Dense retrieval compared with TF-IDF on nDCG@10 over the test queries, reported with confidence intervals |
| G3 | Enforce explicit constraints | Constraint adherence of final recommendations against gold constraints; target 95% or higher |
| G4 | Personalize ranking | Personalization does not materially reduce query relevance, and beats a popularity baseline on held-out next-item HitRate@10 (report the result either way) |
| G5 | Produce grounded responses | Zero recommended product IDs outside the retrieved set; price and attribute mismatch rate reported |
| G6 | Meet atomcamp requirements on time | Notebook, repository, report, and presentation delivered by the dates the program confirms |

## 4. Stakeholders and Users

| Stakeholder | Role | Interest |
| --- | --- | --- |
| Maaz Amjad | Developer and author | Deliver a complete, defensible capstone |
| Mr. Muhammad Umair | Trainer, NLP/LLM track | Feasibility, relevance, methodology, analysis quality |
| Shopper (simulated) | End user | Fast, relevant, explainable results |
| Store owner (future) | Potential customer of a later SaaS version | Measurable discovery improvement |

**Test personas**

- **P1, constraint-driven shopper:** states price, category, or brand limits.
- **P2, descriptive shopper:** vague or attribute-based wording with no explicit limits.
- **P3, returning shopper:** has interaction history for personalization.
- **P4, new shopper:** no history (cold start).

## 5. Scope

**In scope**

- One public product catalog, English only, roughly 10k to 20k products
- LLM query parsing into structured filters plus a semantic query
- Dense retrieval (BAAI/bge-small-en-v1.5 with FAISS) and deterministic filtering
- Lightweight personalization from user interaction history
- Grounded LLM response with automated grounding checks
- Quantitative benchmark with baselines, ablations, and error analysis
- Reproducible Colab/Jupyter notebook and GitHub repository

**Optional (only after the core and evaluation are complete)**

- Streamlit demo interface
- Thin tool wrapper or MCP server exposing search\_products and get\_similar

**Out of scope (future phases)**

Multi-tenant architecture, Shopify/WooCommerce connectors, real-time event streaming, production-scale deployment, LLM fine-tuning, autonomous purchasing, distributed infrastructure, computer vision on product images.

## 6. Assumptions, Constraints, and Dependencies

- **Time:** 1 to 2 weeks of build time; the proposal must be approved before full implementation.
- **Compute:** Google Colab free tier or a local machine; no paid infrastructure.
- **LLM API:** one provider (OpenAI or Google Gemini) selected before implementation, with a spending cap and awareness of rate limits.
- **Data:** public dataset with a license that permits academic use. Price coverage must be profiled before committing to a dataset.
- **Individual project:** assumed unless the program says otherwise.
- **Dates:** the program PDF gives only the proposal deadline (3 October 2026). Final submission and presentation dates must be confirmed.

## 7. Functional Requirements

Priority uses MoSCoW: **M** Must, **S** Should, **C** Could.

### 7.1 Data and catalog

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-D1 | Load a public product dataset and convert it to a canonical schema (product\_id, title, brand, category, price, description, features, popularity count) | M |
| FR-D2 | Clean the data: remove duplicates, empty titles or descriptions, and invalid prices, and log the counts removed | M |
| FR-D3 | Document and implement the missing-price policy: such products are excluded only when a price filter is active | M |
| FR-D4 | Build a product text field for embedding from title, brand, category, features, and description | M |
| FR-D5 | Load user interaction data and keep users with at least 5 interactions for personalization experiments | S |
| FR-D6 | Persist the processed catalog and embeddings to disk so runs are reproducible | M |

### 7.2 Query understanding

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-Q1 | Parse a natural-language query into a structured object: semantic\_query, max\_price, min\_price, category, brand | M |
| FR-Q2 | Restrict category and brand values to those present in the catalog; invalid values are dropped and left to semantic search | M |
| FR-Q3 | Validate parser output against a schema; retry once on failure, then fall back to semantic-only search | M |
| FR-Q4 | Run the parser deterministically (temperature 0) and cache its outputs | S |
| FR-Q5 | Support simple exclusions such as 'not leather' **\[Refinement\]** | C |

### 7.3 Retrieval

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-R1 | Encode the semantic query with BGE and retrieve candidates by cosine similarity | M |
| FR-R2 | Return ranked candidates with their scores | M |
| FR-R3 | Provide baselines: TF-IDF (M), unfiltered dense search (M), BM25 (C) | M |
| FR-R4 | Ensure enough candidates survive filtering, via filter-first search or adaptive over-fetch **\[Refinement\]** | M |

### 7.4 Filtering

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-F1 | Apply hard constraints (price, category, brand) with deterministic Python logic | M |
| FR-F2 | When constraints remove all results, say so and suggest which constraint to relax | M |
| FR-F3 | Treat descriptive terms (for example 'breathable', 'rainy weather') as semantic signals only, never as hard filters | M |

### 7.5 Personalization and ranking

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-P1 | Build a user preference vector from interaction history | M |
| FR-P2 | Compute the final score as a weighted sum of normalized semantic, personal, and popularity scores | M |
| FR-P3 | For users with no history, fall back to semantic relevance (and optionally popularity) | M |
| FR-P4 | Tune weights on a dev split of queries, or fix them in advance and report a sensitivity analysis **\[Refinement\]** | S |

### 7.6 Recommendation and response generation

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-G1 | Return the top 3 to 5 products | M |
| FR-G2 | Generate a concise recommendation or comparison using only retrieved catalog fields | M |
| FR-G3 | Have the LLM return structured output (product IDs with a reason each) that the system renders as text **\[Refinement\]** | S |
| FR-G4 | Run an automated grounding validator: IDs must be in the retrieved set, and prices and brands must match the catalog | M |
| FR-G5 | Explain clearly when no suitable product exists | M |

### 7.7 Evaluation

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-E1 | Author a benchmark of at least 25 stratified queries with gold constraints annotated | M |
| FR-E2 | Produce relevance labels by pooling (graded 0/1/2) and manually verify a sample **\[Refinement\]** | M |
| FR-E3 | Compute Precision@K, pooled Recall@K, NDCG@K, constraint adherence, parse F1, grounding rate, and latency | M |
| FR-E4 | Run an ablation ladder across systems S1 to S6 (defined in the Technical Specification) **\[Refinement\]** | M |
| FR-E5 | Run personalization experiments A (query relevance) and B (held-out next item) **\[Refinement\]** | S |
| FR-E6 | Report bootstrap confidence intervals | S |
| FR-E7 | Perform error analysis across the categories listed in the proposal | M |
| FR-E8 | Split queries into dev and test sets so tuning never uses reported queries | S |

### 7.8 Interface

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-U1 | The notebook runs end to end reproducibly from a clean environment | M |
| FR-U2 | Streamlit demo with a query box, user selector, extracted-filter display, results, and explanation | C |
| FR-U3 | Tool or MCP wrapper exposing search\_products and get\_similar | C |

## 8. Non-Functional Requirements

| ID | Area | Requirement |
| --- | --- | --- |
| NFR-1 | Performance | Retrieval, filtering, and ranking p95 under 500 ms at 20k products on CPU; end-to-end p95 under 6 s, dominated by LLM calls (targets, to be measured) |
| NFR-2 | Reproducibility | Fixed random seeds, pinned library versions, pinned LLM model name, cached LLM outputs, saved result files |
| NFR-3 | Cost | A spending cap set in advance; no LLM pass over the whole catalog |
| NFR-4 | Reliability | Graceful degradation: parser failure falls back to semantic-only search, and LLM generation failure falls back to a template answer |
| NFR-5 | Security and privacy | API keys kept in secrets, never in the repo; no personal data stored; product text treated as untrusted data inside prompts; dataset license respected |
| NFR-6 | Maintainability | Modular code with type hints and docstrings; unit tests for filters, metric functions, and the grounding validator |
| NFR-7 | Explainability | Each recommendation exposes its score components and which constraints it satisfied |
| NFR-8 | Usability | Generated answers are concise (about 120 words or fewer) and mention only verified attributes |

## 9. Data Requirements

| Data | Required fields | Source and notes |
| --- | --- | --- |
| Product catalog | product\_id, title, brand, category, price, description, features | Candidate: a single-category subset of Amazon Reviews 2023 (McAuley Lab). Verify availability, license, and price coverage before committing |
| Interactions | user\_id, product\_id, timestamp (and rating if available) | Same source; users with 5 or more interactions |
| Query benchmark | query text, gold constraints, stratum, dev/test flag | Authored by Maaz |
| Relevance labels | query\_id, product\_id, grade 0/1/2 | Produced by pooling plus LLM judge plus manual verification |

**Dataset selection criteria:** price present for most retained items, usable descriptions, interaction data linked to the same items, and a license that allows academic use. Profile price coverage and category counts before fixing the choice.

## 10. Evaluation Requirements and Acceptance Criteria

| Criterion | Evidence |
| --- | --- |
| Baselines and the proposed pipeline run on all benchmark queries | Results table with all metrics per system |
| Constraint handling is measured | Constraint adherence and parse F1 reported |
| Each stage's contribution is isolated | Ablation table (S1 to S6) with confidence intervals |
| Personalization is evaluated honestly | Experiments A and B reported, including neutral or negative findings |
| Responses are grounded | Grounding validator results across all queries |
| Failures are understood | Error analysis with examples per category |
| Work is reproducible | README plus a notebook that runs from scratch |

**Minimum viable acceptance:** S1, S3, and S5 run on all queries with a metrics table, constraint adherence measured, the grounding validator run, an error analysis written, and a working demonstration (notebook or Streamlit).

## 11. Deliverables and Mapping to atomcamp Expectations

| Deliverable | atomcamp expectation it supports |
| --- | --- |
| Project proposal (submitted) | Clear problem identification |
| This Requirements Document | Clear problem identification |
| Technical Specification | Appropriate methodology |
| Colab/Jupyter notebook and GitHub repository with README | Sound technical implementation |
| Final report with results, ablations, and error analysis | Logical analysis and conclusions |
| Presentation slides and a 2-minute demo video (backup) | Final project presentation |

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dataset has poor price coverage or no usable interactions | Filters and personalization cannot be evaluated | Profile early; switch category or dataset on day 1 to 2 |
| Relevance labeling takes too long | Evaluation incomplete | Pool only the top 10 per system; cap queries at 25 to 40; label with an LLM judge and verify a sample |
| Filter-after-retrieval returns too few results | Poor recall under constraints | Filter-first search or adaptive over-fetch (FR-R4) |
| LLM parser mis-maps categories | Wrong products removed | Restrict to catalog values, validate with a schema, drop invalid filters |
| Personalization shows no benefit | Weak result | Report honestly; run experiment B; keep it as an analyzed finding |
| LLM API cost or rate limits | Delays | Cache outputs, spending cap, small model for parsing |
| Trainer requests major revisions | Rework | Keep the pipeline modular; core retrieval and evaluation survive most scope changes |
| Time overrun | Missed deadline | Follow the cut order: MCP, Streamlit, popularity term, experiment B |

## 13. Milestones (10-day build plan)

| Day | Milestone |
| --- | --- |
| 1 | Approval confirmed, dataset chosen and profiled, LLM provider fixed |
| 2 to 3 | Canonical schema, cleaning, EDA, embeddings, index, baselines |
| 4 | Query parser, category mapping, filtering, filter-aware retrieval |
| 5 | Benchmark queries, gold constraints, evaluation harness |
| 6 | Pooling and relevance labeling |
| 7 | Personalization, experiments A and B |
| 8 | Grounded generation and grounding validator |
| 9 | Ablations, confidence intervals, error analysis |
| 10 | README, report, slides, demo video, optional Streamlit |
