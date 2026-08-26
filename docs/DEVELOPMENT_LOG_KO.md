# AI DATA EXPERT 개발 기록

기준일: **2026-08-26**

## 현재 기준선

- Version: `V6.1_CANDIDATE`
- Status: `CANDIDATE_NOT_PROMOTED`
- Freeze-time regression: `44/44 PASS`
- Kaggle MASTER_EVAL V1 proxy: `3835/4000 = 95.875`
- MASTER_EVAL: `35 PASS / 5 REVIEW / 0 FAIL`
- Real Kaggle data coverage: `0/40`
- Freeze integrity: `PASS`

## 버전 흐름

### V1–V3
- Repository Skill packaging
- TaskSpec / Intent Router
- Domain RAG
- Hypothesis/Experiment
- Challenger/Verifier
- PyTorch DL / Pixel CNN
- regression/classification diagnostics

### V4
- partial-label semantics
- ID/row-order proxy guard
- group/run/time split inference
- Hybrid Domain RAG + structured fact injection
- semantic notebook validation
- Windows UTF-8 / install fail-fast

### V5
- Shared Evidence Store
- Argument Ledger
- Time-Series specialist
- chronological validation
- train-only scaling
- persistence baseline
- SimpleRNN/LSTM comparison
- modality-specific verifier

### V6
- CompetitionSpec
- Competition Planner / Verifier
- competition metric/direction/submission contract
- Human-friendly Renderer
- Kaggle MASTER_EVAL harness

V6 최초 후보는 내부 40문제에서 높은 점수를 냈지만 기존 전체 회귀에서 2개 기능 퇴보가 발견되어 Release하지 않았습니다.

### V6.1
수정 범위:
- explicit forecast negation
- explicit forecast horizon recognition

수정 후 기존 회귀를 모두 다시 통과한 상태에서 Core Freeze 후 MASTER_EVAL을 수행했습니다.

## Freeze-time Regression

| Suite | 결과 |
|---|---:|
| inherited V3 | 22/22 PASS |
| V4 | 8/8 PASS |
| V5 | 7/7 PASS |
| V6 | 5/5 PASS |
| V6.1 | 2/2 PASS |

## Kaggle MASTER_EVAL V1

내부 proxy benchmark 결과:

- strict composite KPI: `95.875`
- reasoning contract KPI without baseline superiority penalty: `98.375`
- proxy model improvement: `30/40 = 75%`
- PASS 35 / REVIEW 5 / FAIL 0

REVIEW:
- Bike Sharing Demand — group/time validation conflict
- Spaceship Titanic — expected stratified vs conservative group-aware choice
- IEEE Fraud Detection — group/time validation conflict
- M5 Forecasting — exact WRMSSE runtime unavailable
- COVID Week 5 — exact weighted pinball runtime unavailable

## 다음 P0

1. Time + Group composite validation planner
2. exact competition metric adapters
3. rolling-origin + multi-seed + seasonal baseline
4. real Kaggle/MLE-bench data execution
5. harder Vision source/group/OOD proxy

## Release Policy

```text
Development
  ↓
Regression
  ↓
Freeze
  ↓
New evaluation
  ↓
Failure preserved
  ↓
Next version only
```

- Freeze 이후 발견한 실패를 같은 Freeze 결과에 몰래 반영하지 않음
- self-reported PASS보다 Test/Evidence를 우선
- 높은 benchmark score보다 regression integrity를 우선
- V6 실패 증거를 보존하고 V6.1을 별도 candidate로 유지
