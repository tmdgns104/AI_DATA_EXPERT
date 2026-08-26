# AI DATA EXPERT 개발 기록

기준일: **2026-08-26**

## 현재 Release Baseline

- Version: `AI_DATA_EXPERT_CODEX_SKILL_V4`
- Core status: `FROZEN_AFTER_TARGETED_SEALED_SIMULATION`
- Distribution: `V4-clone-ready`
- V4 improvement tests: `8/8 PASS`
- inherited V3 regression: `22/22 기능 PASS`
- targeted sealed: `96/96 PASS`
- clone-ready runtime smoke: `PASS`
- isolated V4 Notebook E2E rerun: `PASS`

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
| `verify_install.py` | 설치 후 Runtime/Route/Verifier smoke |
| `examples/generate_demo_data.py` | clone 후 deterministic demo data 생성 |

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

### V4 Core
- partial label semantics
- ID/row-order proxy exclusion
- group/run inference
- Hybrid RAG + structured fact injection
- semantic notebook validation
- Windows UTF-8 / install fail-fast
- V3 sealed failures regression fixes

### V4 Clone-Ready Distribution Hardening

V4 분석 Core의 Benchmark 기준선은 유지하고 배포/설치 계층만 보강했습니다.

- GitHub clone/ZIP 후 `setup_windows.bat` 한 번으로 `.venv` 생성
- Python/`py -3` bootstrap 탐지
- Python 3.11+ 확인
- `torchvision`, Pillow, PyYAML 등 실사용 누락 의존성 보완
- Jupyter/IPython 디렉터리를 Repository 내부로 지정해 Windows 사용자 프로필 권한 문제 완화
- setup 중 deterministic demo CSV 생성
- `verify_install.py`에서 dependency → Skill → Runtime import → 실제 분류 Expert 실행 → Route/Verifier 확인
- 실패 시 `Setup complete`를 출력하지 않고 non-zero 종료
- `run_demo_without_codex.bat`가 `.venv`가 없으면 setup을 먼저 실행
- 실제 DNN 희귀 클래스 사용 결과를 반영해 Rare-event `REVIEW` 규칙 추가
- 실제 Vision 사용 결과를 반영해 PyTorch `state_dict`/safe reload 규칙 추가

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

- Freeze 이후 발견한 실패를 같은 버전 Core에 몰래 수정하지 않음
- self-reported PASS를 신뢰하지 않고 Test/Evidence로 판정
- 다른 Holdout 점수를 직접 %p 비교하지 않음
- 설치/문서/의존성 hardening과 분석 Core 변경을 분리 기록

## 실제 사용에서 확인한 장점

- 극단적 불균형에서 Accuracy 과신을 막고 Macro-F1 / Balanced Accuracy / defect Recall / PR-AUC로 전환
- 불량 9건처럼 표본이 너무 적을 때 운영 판정을 `REVIEW`로 제한
- 실제 Pixel Vision에서 image overlap 확인, checkpoint 선택, 모델 저장/재로딩까지 검증
- 코드 실행 성공과 Production 적용 가능성을 분리

## 다음 버전 후보 — 아직 V4 Core에 미구현

### P0
- Shared Evidence Store: Vision / DL / ML Expert가 동일 실행 상태를 공유
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
- output/log UTF-8 일관화

## Release Policy

`main`의 V4 Core는 현재 재현 가능한 분석 기준선으로 유지합니다. 배포 hardening은 별도 release 기록으로 남기고, 다음 분석 기능 변경은 V5에서 진행합니다.
