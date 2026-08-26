# Repository Agent Instructions — AI Data Expert V4

Use `.agents/skills/ai-data-expert/SKILL.md` for data analysis, ML/DL, Vision, time-series, Big Data, MLOps, causal analysis, and Jupyter notebook tasks.

Operating flow:
`Problem -> TaskSpec/DataGuard -> Hybrid Domain RAG -> Intent Router -> Experts -> Hypothesis/Experiment -> Challenger -> Semantic Verifier -> Artifact`

Rules:
- Preserve original question cells in notebook assignments.
- Target-missing rows are unlabeled prediction rows, never a new target class.
- Detect and exclude high-cardinality ID/row-order proxies unless domain evidence explicitly justifies them.
- Prefer entity/run/time-isolated validation when repeated structure is detected; verify zero group overlap.
- Never use final Test for model/threshold/hyperparameter selection.
- Retrieved domain facts must be source-traced and applied to TaskSpec/feature eligibility when structured evidence supports it.
- Never interpret feature importance as causality.
- Do not auto-retrain from drift alone.
- Runtime `FAIL` cannot be presented as completed; `REVIEW` requires caveats.
- Execute and semantically validate notebooks before completion.
- V3 remains frozen; V4 changes live only in this repository copy.
