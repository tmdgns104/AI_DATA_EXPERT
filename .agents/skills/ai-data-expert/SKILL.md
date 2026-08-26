---
name: ai-data-expert
description: Use for CSV/data analysis, EDA, statistics, machine learning, deep learning, computer vision, time-series, competition-aware modeling, big-data, MLOps, causal questions, or Jupyter notebook tasks that need evidence-backed expert judgment and executable validation.
---

# AI Data Expert V6.5 Frozen Candidate

Use this skill whenever the task is primarily data/AI analysis or a Jupyter data-science assignment.

## Principle

Do not begin with a model or a label. The active path is:

`Observe -> TaskSpec/Data Guard -> Domain RAG -> Intent/Modality -> Experts -> Shared Evidence/Argument -> Experiment -> Challenger -> Verifier -> Human-friendly Output`

Execution alone is not enough. Preserve uncertainty internally, but render the final notebook in simple human language.

## 1. Inspect before deciding

For notebooks:

```bash
python .agents/skills/ai-data-expert/scripts/inspect_notebook.py <question.ipynb>
```

Identify observation unit, target, prediction time, available features, group/entity structure, timestamp, metric, validation requirement, business cost, deployment constraints, and competition contract when relevant.

Unknown fields stay unknown. Do not silently invent them.

For data analysis, prefer this visible reasoning order:

`데이터 확인 -> 이상/구조 발견 -> 근거 확인 -> 문제 유형 판단 -> 검증 방법 선택 -> baseline -> model -> Test 해석`

## 2. Run the expert harness

```bash
python .agents/skills/ai-data-expert/scripts/run_expert.py \
  --csv <data.csv> \
  --task "<request>" \
  --target <target-if-known> \
  --out outputs/expert_context.json
```

Useful flags include `--prediction-time`, `--business-cost`, `--domain-path`, `--modality time-series`, `--timestamp-col`, `--horizon`, `--monitoring`, `--deployment`, `--streaming`, and `--image-npz`.

Internally inspect `task_spec`, `data_guard`, `domain_context`, `routing`, `shared_evidence`, `argument_ledger`, `challenger`, and `verification`. Never promote a real `FAIL`.

## 3. Mandatory guards

- Target-missing rows are unlabeled rows, not a new class.
- Detect unique/monotonic IDs, direct target copies, affine target proxies, and post-outcome features.
- Repeated entities can invalidate random-row validation, including unfamiliar aliases such as account/crew/family/merchant/cohort/unit/ref/token patterns.
- Time validation must inspect not only order/duplicates but cadence breaks.
- Group and Time can both matter; use combined sensitivity/temporal-group validation when appropriate.
- Final Test cannot select model, features, threshold, lag, sequence length, or hyperparameters.
- Baselines compete in Validation. A complex model is not champion merely because it is complex.
- Under imbalance, accuracy alone is insufficient.
- Predictive feature importance is not causal evidence.

## 4. EDA must change the next decision

EDA is not decoration. Observations should lead to a decision or an experiment.

Examples:
- strong lag-1 correlation -> persistence baseline is required
- strong lag-96 correlation on 15-minute data -> daily seasonal baseline/input-window sensitivity is worth checking
- repeated entity rows -> group-aware validation
- timestamp gaps -> cadence integrity review
- minority support is tiny -> threshold/recall stability cannot be claimed from one split

If an input length such as 32 steps is not given by the assignment, write that it is an experiment condition. If practical, compare a small set of candidate lengths on Validation rather than calling one value optimal.

## 5. Validation selects; Test reports

Use Validation to choose the model/threshold/checkpoint/feature set/sequence length. Open final Test after selection.

The final explanation should center on Test results while mentioning that the choice was made on Validation.

When metrics disagree, report the disagreement. Do not write “best on every metric” unless that is actually true.

## 6. Fair model comparison

Keep data, split, preprocessing and training budget comparable where possible, but do not imply equal model complexity merely because hidden size is the same. Parameter count/resource differences may be noted when they matter.

## 7. Time-series

- inspect timestamp parsing, order, duplicates, dominant cadence and cadence breaks
- use chronological/rolling validation for forecasting
- fit preprocessing on Train only
- compare persistence and meaningful seasonal-naive baselines
- explicit `do not forecast` suppresses forecast-only requirements
- explicit horizons such as `next 24 hours` must not become UNKNOWN
- for the Steel Industry exercise, an unspecified one-step/sequence length is an explicit exercise assumption, not a discovered optimum

`solve_timeseries_rnn_v5.py` remains a dataset-specific exercise solver, not a universal forecasting engine.

## 8. Competition-aware tasks

Preserve target, official metric, direction, validation logic and submission contract. Generic local metrics may supplement but cannot replace the official metric.

If exact scoring needs unavailable hierarchy/weights/private artifacts, mark the reproduction approximate rather than pretending it is the leaderboard score.

## 9. Domain RAG

Retrieved material is evidence, not truth. Keep provenance, reject target/domain mismatches, and respect domain negation such as `not manufacturing`, `제조와 무관`, or requests to exclude factory context.

## 10. Human-friendly Notebook output

Follow `.agents/skills/ai-data-expert/OUTPUT_STYLE_CONTRACT_KO.md`.

Internal reasoning may use Argument Ledger, Evidence IDs and Verifier states. The final notebook should normally not show those terms. Prefer short, natural explanations such as `확인해봄`, `정했음`, `비교했음` when that matches the user's requested style.

## 11. Version discipline

- Historical freezes are provenance; never rewrite them to improve a score.
- V6.4 was frozen before its independent 200-case holdout and scored 80/200 on that harder holdout.
- V6.5 fixes the resulting blind spots: unseen group aliases, cadence breaks, direct/affine target leakage, plus V6.4 RAG negation behavior.
- V6.5 must be frozen before interpreting new benchmark outcomes.
- Kaggle fallback-data experiments are not Kaggle leaderboard scores and must be labeled honestly.
