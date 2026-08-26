---
name: ai-data-expert
description: Use for CSV/data analysis, EDA, statistics, machine learning, deep learning, computer vision, time-series, competition-aware modeling, big-data, MLOps, causal questions, or Jupyter notebook tasks that need evidence-backed expert judgment and executable validation.
---

# AI Data Expert V6.1 Candidate

Use this skill whenever the task is primarily data/AI analysis or a Jupyter data-science assignment.

## Principle

Do not begin with a model. The active V6.1 path is:

`TaskSpec/DataGuard -> Hybrid Domain RAG -> Intent/Modality Router -> Experts -> Shared Evidence/Argument Ledger -> Hypothesis/Experiment -> Challenger -> Modality/Competition Verifier -> Human Output`

A result that merely executes is not sufficient. Preserve uncertainty and return `REVIEW`/`FAIL` when required context is missing.

## 1. Inspect the request and local inputs

For notebooks:

```bash
python .agents/skills/ai-data-expert/scripts/inspect_notebook.py <question.ipynb>
```

Identify or infer:
- observation unit
- target
- target business meaning
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
- if competition task: exact competition metric, direction, validation contract, and submission format

Unknown fields must remain `UNKNOWN`; do not silently invent them.

## 2. Run the active expert harness before authoring

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
- `--modality time-series`
- `--timestamp-col <column>`
- `--horizon <horizon>`
- `--monitoring --existing-model`
- `--deployment`
- `--streaming --size-gb <GB>`
- `--image-npz <images.npz>` for actual pixel CNN simulation/training

Read these blocks before writing the final answer:
- `task_spec`
- `data_guard`
- `domain_context`
- `routing`
- expert `DECIDE/RISKS/CONFIDENCE`
- `shared_evidence`
- `argument_ledger`
- `hypotheses`
- `experiment_evidence`
- `challenger`
- `verification`

Never promote a `FAIL`. Surface `REVIEW` caveats.

## 3. Mandatory data/validation guards

- Target-missing rows are unlabeled rows, not a new target class.
- Exclude obvious ID, serial, UUID, and near-monotonic row-order proxies unless evidence justifies them.
- Observation unit comes before split choice.
- Repeated entities can invalidate random-row split.
- Time structure can invalidate shuffled validation.
- Group and Time may both matter. Do not blindly let one precedence rule override the other.
- Final Test cannot be used for model, feature, threshold, or hyperparameter selection.
- Complex models must beat defensible simple baselines before claiming improvement.
- Under imbalance, accuracy alone is insufficient; include macro/balanced/per-class or event-focused metrics as appropriate.
- Failure analysis must inspect worst rows/classes/segments, not only global averages.
- Predictive feature importance is not a causal effect.

## 4. Notebook tasks

For ordinary tabular regression/classification:

```bash
python .agents/skills/ai-data-expert/scripts/solve_notebook.py \
  --input <question.ipynb> \
  --data <data.csv> \
  --output <answer.ipynb>
```

Preserve original question cells. Use Train/Validation/Test correctly and run semantic validation.

```bash
python .agents/skills/ai-data-expert/scripts/validate_notebook.py <answer.ipynb>
```

## 5. Time-Series V5/V6.1 contract

For time-series analysis:
- parse timestamp and inspect duplicates/order/dominant interval
- use chronological or rolling-origin validation for forecast tasks
- fit scaling/preprocessing on Train only
- compare against persistence/naive and preferably seasonal-naive baselines
- select model/checkpoint on Validation, then open final Test once
- record assumptions in Shared Evidence and Argument Ledger

Explicit forecast negation must be respected:

```text
historical time-series analysis only; do not forecast
```

must not trigger forecast-only requirements.

Explicit horizon must be preserved:

```text
forecast the next 24 hours
horizon=24h
```

must not be downgraded to `UNKNOWN`.

`solve_timeseries_rnn_v5.py` is currently a dataset-specific notebook solver for the included Steel Industry exercise. Do not present it as a universal forecasting solver.

## 6. Competition-aware tasks

V6 adds CompetitionSpec/Planner/Verifier behavior.

Before modeling, fix:
- competition target
- official selection metric
- metric direction
- validation expectation
- leakage risks
- submission mode

Generic local metrics may be diagnostic only. They cannot replace the competition metric.

If exact competition scoring requires unavailable artifacts such as hierarchy, scale weights, quantile weights, or private evaluation logic, mark runtime as approximate and keep the result at `REVIEW` rather than pretending exact reproduction.

The included 40-competition MASTER_EVAL is an internal proxy benchmark. It is not a Kaggle leaderboard result.

## 7. Hybrid Domain RAG

Place project/domain evidence in `domain_knowledge/` or pass `--domain-path`.

Retrieval combines lexical/vector/metadata signals. Structured facts can refine prediction time, group id, feature eligibility, and business cost only when source-traced evidence supports them.

Retrieved material is evidence, not universal truth. Conflicting or weak evidence must reduce confidence.

## 8. Argument Ledger / Challenger

Major decisions should be expressible as:

`Question -> competing hypotheses -> required evidence -> observation -> counterargument -> decision -> status -> next question`

Useful statuses:
- `OPEN`
- `SUPPORTED`
- `REJECTED`
- `INCONCLUSIVE`
- `BLOCKED`

`SUPPORTED` means survived current evidence, not proven forever.

The Challenger should attempt to falsify:
- leakage/proxy?
- wrong split?
- baseline missing?
- test reused?
- minority/segment failure hidden?
- probability/threshold unstable?
- causal overclaim?
- forecast horizon invented?
- competition metric replaced?
- domain evidence contradicted?

## 9. Runtime status

- `PASS`: no critical contract violation found in current evidence
- `REVIEW`: analysis can proceed but material uncertainty/assumption remains
- `FAIL`: execution or contract violation blocks promotion

`REVIEW` is a valid output and must not be cosmetically upgraded.

## 10. Version discipline

- Historical V3/V4/V5/V6 freezes are provenance and must not be rewritten.
- V6 was not promoted because inherited regression failures were discovered after its first benchmark run.
- V6.1 repaired forecast negation and explicit horizon handling, then was frozen after the recorded 44/44 regression pass.
- Current main baseline is `V6.1_CANDIDATE`, not Production Release.
- New evaluation failures belong to the next version; do not mutate frozen evidence to improve the score.
