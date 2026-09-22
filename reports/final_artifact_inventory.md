# AI Customer Support Agent: Final Artifact Inventory

---

## 1. Executive Summary & Core Reports
- **[`reports/final_report.md`](file:///e:/AI-Customer-Support-Agent/reports/final_report.md)**: The primary 6-page comprehensive technical report covering problem framing, brand selection, system architecture, baselines vs. semantic reranking, grounded response generation, LLM-as-a-judge evaluation, failure modes, and one-more-week engineering plan.
- **[`reports/submission_checklist.md`](file:///e:/AI-Customer-Support-Agent/reports/submission_checklist.md)**: Complete audit checklist verifying all required assignment criteria.
- **[`decision_log.md`](file:///e:/AI-Customer-Support-Agent/decision_log.md)**: Comprehensive chronological log of 37 non-obvious engineering decisions made across Phases 0–10.
- **[`README.md`](file:///e:/AI-Customer-Support-Agent/README.md)**: Main project documentation with end-to-end architecture diagrams, quick-start guides, reproduction commands, and baseline performance tables.

---

## 2. Evaluation Datasets & Benchmarks
- **[`data/golden/golden_inputs.jsonl`](file:///e:/AI-Customer-Support-Agent/data/golden/golden_inputs.jsonl)**: 200 held-out customer queries from post-cutoff timestamps, completely isolated from training.
- **[`data/golden/golden_labels.jsonl`](file:///e:/AI-Customer-Support-Agent/data/golden/golden_labels.jsonl)**: Ground-truth human annotations for the 200 golden queries (intent and triage routing).
- **[`data/evaluation/human_eval_set.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_eval_set.jsonl)**: Stratified 50-example subset balancing all 11 intents and routing decisions for human evaluation.
- **[`data/evaluation/human_annotations_template.jsonl`](file:///e:/AI-Customer-Support-Agent/data/evaluation/human_annotations_template.jsonl)**: Ready-to-use human reviewer annotation template with full 1–5 scoring schema.

---

## 3. Evaluation Scripts & Reproduction Tools
- **[`scripts/evaluate_baselines.py`](file:///e:/AI-Customer-Support-Agent/scripts/evaluate_baselines.py)**: Evaluates Baseline 1 (Majority Class) and Baseline 2 (TF-IDF Intent Classifier + TF-IDF Retriever) on the 200 golden queries.
- **[`scripts/build_semantic_index.py`](file:///e:/AI-Customer-Support-Agent/scripts/build_semantic_index.py)**: Encodes 8,000 pre-cutoff development support interactions into dense MiniLM embeddings and builds `faiss.IndexFlatIP`.
- **[`scripts/evaluate_semantic_retrieval.py`](file:///e:/AI-Customer-Support-Agent/scripts/evaluate_semantic_retrieval.py)**: Evaluates pure semantic retrieval (Top-k=5) against the 200 golden queries.
- **[`scripts/evaluate_intent_aware_retrieval.py`](file:///e:/AI-Customer-Support-Agent/scripts/evaluate_intent_aware_retrieval.py)**: Evaluates intent-aware candidate reranking with soft compatibility bonuses and confidence fallback gates.
- **[`scripts/evaluate_phase8_agent.py`](file:///e:/AI-Customer-Support-Agent/scripts/evaluate_phase8_agent.py)**: Executes the full end-to-end AI Support Agent across 200 golden queries, computing confusion matrices and dangerous false auto-handles.
- **[`scripts/run_llm_judge.py`](file:///e:/AI-Customer-Support-Agent/scripts/run_llm_judge.py)**: Runs multi-dimensional LLM-as-a-Judge evaluation across all 200 generated responses.
- **[`scripts/build_human_eval_set.py`](file:///e:/AI-Customer-Support-Agent/scripts/build_human_eval_set.py)**: Stratified sampler generating the 50-example human review benchmark.
- **[`scripts/import_human_annotations.py`](file:///e:/AI-Customer-Support-Agent/scripts/import_human_annotations.py)**: Validates human reviewer annotations against schema constraints without fabricating missing data.
- **[`scripts/analyze_judge_human_agreement.py`](file:///e:/AI-Customer-Support-Agent/scripts/analyze_judge_human_agreement.py)**: Calculates exact match %, MAD, weighted Cohen's Kappa, and Pearson correlation between LLM Judge and human reviewers.

---

## 4. Phase-by-Phase Technical Reports
- **Phase 1**: [`reports/dataset_schema.md`](file:///e:/AI-Customer-Support-Agent/reports/dataset_schema.md), [`reports/speaker_analysis.md`](file:///e:/AI-Customer-Support-Agent/reports/speaker_analysis.md), [`reports/brand_candidates.md`](file:///e:/AI-Customer-Support-Agent/reports/brand_candidates.md)
- **Phase 2**: [`reports/amazonhelp_response_patterns.md`](file:///e:/AI-Customer-Support-Agent/reports/amazonhelp_response_patterns.md)
- **Phase 3**: [`reports/intent_taxonomy.md`](file:///e:/AI-Customer-Support-Agent/reports/intent_taxonomy.md), [`reports/intent_boundary_analysis.md`](file:///e:/AI-Customer-Support-Agent/reports/intent_boundary_analysis.md)
- **Phase 4**: [`reports/golden_set_creation_report.md`](file:///e:/AI-Customer-Support-Agent/reports/golden_set_creation_report.md)
- **Phase 5**: [`reports/baseline_evaluation_report.md`](file:///e:/AI-Customer-Support-Agent/reports/baseline_evaluation_report.md)
- **Phase 6**: [`reports/semantic_retrieval_results/summary.md`](file:///e:/AI-Customer-Support-Agent/reports/semantic_retrieval_results/summary.md)
- **Phase 7**: [`reports/intent_aware_retrieval_results/summary.md`](file:///e:/AI-Customer-Support-Agent/reports/intent_aware_retrieval_results/summary.md)
- **Phase 8**: [`reports/phase8_results/summary.md`](file:///e:/AI-Customer-Support-Agent/reports/phase8_results/summary.md), [`reports/phase8_results/metrics.json`](file:///e:/AI-Customer-Support-Agent/reports/phase8_results/metrics.json)
- **Phase 9**: [`reports/judge/judge_summary.md`](file:///e:/AI-Customer-Support-Agent/reports/judge/judge_summary.md), [`reports/judge/judge_metrics.json`](file:///e:/AI-Customer-Support-Agent/reports/judge/judge_metrics.json), [`reports/human/agreement_summary.md`](file:///e:/AI-Customer-Support-Agent/reports/human/agreement_summary.md)
- **Phase 10**: [`reports/final_report.md`](file:///e:/AI-Customer-Support-Agent/reports/final_report.md)

---

## 5. Core Source Code Architecture (`src/`)
- **`src/preprocessing/`**: PII sanitization and conversation reconstruction graph traversal.
- **`src/intents/`**: 11-intent taxonomy schemas and TF-IDF logistic regression classifier.
- **`src/retrieval/`**: Sentence-transformers MiniLM dense embedding generator, FAISS vector index, and intent-aware soft reranker.
- **`src/generation/`**: Evidence selection & heuristic quality scoring, prompt construction, LLM provider abstraction (`MockLLMProvider`, `GeminiLLMProvider`, `OpenAILLMProvider`), and grounding guardrails.
- **`src/escalation/`**: Deterministic rule-based triage engine with mandatory escalation logic.
- **`src/pipeline/`**: Unified `AISupportAgent` end-to-end customer interaction orchestrator.
- **`src/evaluation/`**: Baseline systems, LLM Judge (`MockLLMJudge`, `GeminiLLMJudge`, `OpenAILLMJudge`), and mathematical agreement calculator (Weighted Cohen's Kappa, MAD, Pearson $r$).

---

## 6. Unit Test Suite (`tests/`)
- Total of **68 passing unit tests** verifying conversation reconstruction, PII masking, taxonomy validation, baseline classification, vector indexing, reranking logic, response schema, guardrails, triage rules, LLM judge scoring, and agreement mathematics.
