---
name: ai-data-expert
description: Use for CSV/data analysis, EDA, statistics, machine learning, deep learning, computer vision, time-series, big-data, MLOps, causal questions, or Jupyter notebook tasks that need evidence-backed expert judgment and executable validation.
---

# AI Data Expert V4

Use this skill whenever the task is primarily data/AI analysis or a Jupyter data-science assignment.

## Principle

Do not begin with a model. The V4 runtime follows:

`TaskSpec/DataGuard -> Hybrid Domain RAG -> Intent Router -> Experts -> Hypothesis/Experiment -> Challenger -> Semantic Verifier`

A result that merely executes is not sufficient. Preserve uncertainty and return `REVIEW`/`FAIL` when required context is missing.

## 1. Inspect the request and local inputs

For notebooks:

```bash
python .agents/skills/ai-data-expert/scripts/inspect_notebook.py <question.ipynb>
```

Identify or infer:
- observational unit
- target
- prediction time
- features available at prediction time
- entity/group id
- timestamp
- problem type
- validation split
- primary metric
- business error cost
- predictive vs causal question
- deployment constraints

Unknown fields must remain `UNKNOWN`; do not silently invent them.

## 2. Run the V4 expert harness before authoring

```bash
python .agents/skills/ai-data-expert/scripts/run_expert.py \
  --csv <data.csv> \
  --task "<request>" \
  --target <target-if-known> \
  --out outputs/expert_context.json
```

Useful optional flags:
- `--prediction-time "..."`
- `--business-cost "..."`
- `--domain-path <folder-or-file>`
- `--monitoring --existing-model`
- `--deployment`
- `--streaming --size-gb <GB>`
- `--image-npz <images.npz>` for actual pixel CNN simulation/training

Read these result blocks before writing the answer:
- `task_spec`
- `domain_context`
- `routing`
- expert `DECIDE/RISKS/CONFIDENCE`
- `hypotheses`
- `experiment_evidence`
- `challenger`
- `verification`

Never promote a `FAIL` result. Surface `REVIEW` caveats.

## 3. Notebook tasks

For ordinary tabular regression/classification:

```bash
python .agents/skills/ai-data-expert/scripts/solve_notebook.py \
  --input <question.ipynb> \
  --data <data.csv> \
  --output <answer.ipynb>
```

The V4 solver must:
- preserve original question cells
- separate target-missing rows as unlabeled predictions
- remove high-cardinality identifier/row-order proxies
- infer repeated production/entity groups and prefer zero-overlap group validation
- separate Train / Validation / Test
- select model/threshold without using final Test
- include simple baselines
- evaluate failure cases/segments
- regression: residual/uncertainty diagnostic and worst errors
- classification: macro-F1, balanced accuracy, per-class metrics, probability quality, validation threshold review
- keep feature importance separate from causal claims

Always validate:

```bash
python .agents/skills/ai-data-expert/scripts/validate_notebook.py <answer.ipynb>
```

## 4. Mandatory expert guards

- **Observation unit first:** repeated entity rows can invalidate random split.
- **Prediction time:** flag post-outcome or unavailable features.
- **Test isolation:** model/hyperparameter/threshold selection cannot use final Test.
- **Baselines:** complex models must beat defensible simple baselines.
- **Metrics:** match the problem and business loss; accuracy alone is insufficient under imbalance.
- **Calibration/threshold:** probability decisions need probability-quality and threshold review.
- **Failure analysis:** inspect worst rows/segments, not only overall score.
- **Uncertainty/OOD:** surface uncertainty or distribution-shift risk when practical.
- **Causality:** predictive importance is not intervention effect.
- **Deep learning:** when actual inputs are available, require PyTorch execution, small-batch overfit sanity, and best-validation checkpoint evidence.
- **Vision:** define classification/detection/segmentation/anomaly task first; guard product/source leakage and augmentation-label semantics.
- **Forecast:** chronological/rolling validation and naive/seasonal-naive baseline.
- **MLOps:** drift does not automatically authorize retraining; investigate pipeline/process/root cause.
- **Big Data:** choose technology from reliability/scalability/maintainability/SLA constraints, not buzzwords.

## 5. Hybrid Domain RAG

Place project/domain evidence in `domain_knowledge/` or pass `--domain-path`.
Retrieval combines BM25 lexical search, vector similarity, and metadata boosts. If sentence-transformers + FAISS are locally installed, V4 uses them; otherwise it uses an offline vector fallback. Structured JSON facts can update TaskSpec/feature eligibility, and every applied fact must retain its source. Retrieved material is evidence, not universal truth.

## 6. Challenger and Verifier

The Challenger must attempt to falsify the proposed answer:
- leakage/proxy?
- wrong split?
- baseline missing?
- test reused?
- minority/segment failure hidden?
- probability miscalibrated?
- causal overclaim?
- DL pipeline sanity missing?
- business cost/prediction time unknown?

The Runtime Verifier checks the reasoning contract, expert execution, required baselines/metrics, hypothesis/experiment evidence, challenger outcome, and task-specific safety guards.

## 7. Version discipline

Parent Tacit heuristics/sources remain provenance-frozen. V3 changes are layered in new runtime modules. V3 remains frozen. V4 fixes are layered separately and must be frozen before new sealed evaluation.
