# AI DATA EXPERT 개발 기록

기준일: **2026-08-26**

## 현재 Release Baseline

- Version: `AI_DATA_EXPERT_CODEX_SKILL_V4`
- Status: `FROZEN_AFTER_TARGETED_SEALED_SIMULATION`
- V4 improvement tests: `8/8 PASS`
- inherited V3 regression: `22/22 기능 PASS`
- targeted sealed: `96/96 PASS`
- freeze integrity: `PASS`

## V4 핵심 구현 파일

| 파일 | 역할 |
|---|---|
| `data_expert/v4_system.py` | V4 orchestration |
| `data_expert/task_spec_v4.py` | Task contract |
| `data_expert/data_guard_v4.py` | Target/ID/Group/Time guard |
| `data_expert/domain_rag_v4.py` | Hybrid RAG |
| `data_expert/advanced_ml_v4.py` | ML diagnostics |
| `data_expert/dl_engine_v4.py` | DL/Vision runtime |
| `data_expert/hypothesis_engine_v4.py` | Hypothesis / evidence |
| `data_expert/challenger_v4.py` | counter-check |
| `.agents/skills/ai-data-expert/SKILL.md` | Codex skill contract |

## V1 → V4 변경 요약

### V1
- Codex Repository Skill packaging
- Notebook Solver E2E
- 실제 사용에서 classification/Test-selection/Router/Verifier 결함 발견

### V2
- classification/regression split
- validation/test isolation
- Runtime Verifier
- English/negation routing
- Expert exception isolation

### V3
- TaskSpec
- Intent-first router
- Domain RAG
- Hypothesis/Experiment
- Challenger
- actual PyTorch DL
- pixel CNN
- advanced ML metrics

### V4
- partial label semantics
- ID/row-order proxy exclusion
- group/run inference
- Hybrid RAG + structured fact injection
- semantic notebook validation
- Windows UTF-8 / install fail-fast
- V3 sealed failures regression fixes

## Test Strategy

```text
Development Test
  ↓
Regression Test
  ↓
Domain Smoke
  ↓
Freeze
  ↓
New Sealed Holdout
  ↓
Failure preserved
  ↓
Next version only
```

원칙:

- Freeze 이후 발견한 실패를 같은 버전에 몰래 수정하지 않음
- self-reported PASS를 신뢰하지 않고 Test/Evidence로 판정
- 다른 Holdout 점수를 직접 %p 비교하지 않음

## 실사용에서 발견한 V4 이후 후보

아래 항목은 **아직 V4 구현 완료가 아니라 다음 버전 후보**입니다.

### P0
- Shared Evidence Store: Vision / DL / ML Expert가 동일한 실행 상태를 공유
- Modality-specific Semantic Validator: Tabular / Vision / Time-Series 별 계약 분리
- Argument Ledger: 질문-가설-증거-반증-결정-다음 질문 상태 추적

### P1
- 실제 sentence-transformers + FAISS Windows runtime benchmark
- RAG Recall@K / source precision / wrong-fact injection 평가
- Rare-event threshold uncertainty / bootstrap
- cost-aware threshold optimization
- inferred Group provenance / confidence
- label missing mechanism 분석

### P2
- Windows CI
- Notebook renderer/template 분리
- dependency completeness 자동 검사
- output/log UTF-8 일관화

## Release Policy

`main`의 V4는 현재 재현 가능한 기준선으로 유지합니다.
다음 기능 변경은 V5 branch/candidate에서 진행하고, V4의 Freeze/Evaluation 결과는 변경하지 않습니다.
