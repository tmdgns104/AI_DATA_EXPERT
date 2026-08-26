# Repository Agent Instructions — AI Data Expert V6.5 Frozen Candidate

Use `.agents/skills/ai-data-expert/SKILL.md` for data analysis, ML/DL, Vision, time-series, Big Data, MLOps, causal analysis, competition-aware planning, and Jupyter notebook tasks.

Primary flow:
`Problem -> TaskSpec/DataGuard -> Hybrid Domain RAG -> Intent/Modality Router -> Experts -> Shared Evidence/Argument Ledger -> Hypothesis/Experiment -> Challenger -> Modality/Competition Verifier -> Human Output`

Rules:
- Preserve original question cells in notebook assignments.
- Target-missing rows are unlabeled prediction rows, never a new target class.
- Detect and exclude high-cardinality ID/row-order proxies unless domain evidence explicitly justifies them.
- Prefer validation that matches the data-generating process. Group and Time can both matter; do not blindly apply a single precedence rule.
- Never use final Test for model/threshold/hyperparameter selection.
- Complex models must beat defensible baselines before claiming improvement.
- Retrieved domain facts must retain source/provenance and may refine TaskSpec only when evidence supports them.
- Do not interpret feature importance as causality.
- Do not auto-retrain from drift alone.
- For forecasting, explicit negation such as `do not forecast` must suppress forecast-only requirements.
- For forecasting, explicit horizon such as `next 24 hours` or `horizon=24h` must not be downgraded to UNKNOWN.
- Competition tasks must preserve the competition metric/direction/submission contract; generic local metrics may supplement but never replace it.
- If a competition-specific metric requires unavailable hierarchy/weights/artifacts, return APPROX/REVIEW rather than pretending exact reproduction.
- Runtime `FAIL` cannot be presented as completed; `REVIEW` requires caveats.
- Execute and semantically validate notebooks before completion when execution is part of the task.
- V3/V4/V5/V6/V6.1/V6.2/V6.3/V6.4/V6.5 freeze/evaluation evidence is historical provenance. Do not rewrite old freeze results to improve scores.
- Current main baseline is V6.5 Frozen Candidate, not Production Release.

Final notebook output:
- Show `data check -> observation -> reason -> decision -> experiment -> Test interpretation` naturally.
- Do not expose internal Agent terms such as DataGuard/Argument Ledger/Verifier in student-facing notebooks.
- Values not specified by the assignment (sequence length, lag, threshold) are experiment assumptions, not facts.
- Validation selects; final Test reports. If MAE/RMSE/R2 disagree, preserve the tradeoff instead of saying one model won every metric.
- Scope RNN/LSTM comparisons when parameter counts differ.
