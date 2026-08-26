# AI DATA EXPERT

**Codex에서 데이터 분석·ML/DL·Vision·Time-Series 작업을 더 안전하게 수행하기 위한 검증형 Repository Skill + Expert Harness**

- 최신 기준: **V6.1 Candidate**
- 상태: **Candidate / Pre-Production Expert Copilot**
- Core regression: **44/44 PASS at freeze time**
- Kaggle MASTER_EVAL V1 (proxy): **3835/4000 = 95.875**
- MASTER_EVAL status: **35 PASS / 5 REVIEW / 0 FAIL**
- 실제 Kaggle 원본 데이터 평가: **0/40** — Kaggle API 자격증명/원본 데이터가 필요함

> 이 프로젝트의 목표는 높은 점수를 만드는 AutoML이 아니라, **잘못된 문제 정의·누수·부적절한 검증·과장된 결론을 먼저 의심하고 증거로 판단하는 데이터 분석 Agent**를 만드는 것입니다.

## V6.1 핵심 흐름

```text
User / CSV / Notebook / Competition Task
        ↓
TaskSpec V6.1
        ↓
Data Guard
        ↓
Hybrid Domain RAG
        ↓
Intent / Modality Router
        ↓
Data / ML / DL / Vision / Time-Series / BigData / MLOps Experts
        ↓
Shared Evidence + Argument Ledger
        ↓
Hypothesis / Experiment
        ↓
Local Challenger
        ↓
Modality Verifier
        ↓
CompetitionSpec / Competition Planner / Metric Guard (V6)
        ↓
Human-friendly Output
        ↓
PASS / REVIEW / FAIL
```

## V4 → V6.1 주요 변화

### V4 — 데이터 안전 + Hybrid RAG
- Target missing을 새로운 Class로 만들지 않고 labeled / unlabeled 분리
- `_id`, unique serial, row-order proxy 등 식별자/순서 누수 경고 및 제외
- 반복 Entity / Run / Time 구조 탐지와 split 후보 추론
- Final Test를 모델/Threshold 선택에서 격리
- BM25 + Vector + Metadata + Structured Facts 기반 Hybrid Domain RAG
- Domain fact를 prediction time / group / feature eligibility / business cost에 적용
- Semantic Notebook Validator

### V5 — 논증형 분석 + Time-Series specialist
- Shared Evidence Store
- Argument Ledger
- 질문 → 가설 → 증거 → 반증 → 임시결론 → 다음 질문 구조
- Time-Series 전용 routing / verifier
- timestamp integrity, chronological split, train-only scaling
- Persistence baseline + SimpleRNN/LSTM 비교
- 명시되지 않은 forecast horizon은 가정으로 숨기지 않고 REVIEW

### V6 — Competition-aware planning
- CompetitionSpec
- Competition metric / direction / validation / submission contract
- Competition Planner
- Competition Verifier
- generic metric이 대회 metric을 덮어쓰지 못하도록 guard
- probability / class label / continuous submission mode 구분
- complex metric은 원본 competition artifact가 없으면 APPROX/REVIEW로 남김

### V6.1 — 회귀 수정
V6 Freeze 후 기존 회귀에서 2개 문제를 발견해 V6을 Release하지 않았습니다.

- `do not forecast` 같은 명시적 forecast negation 처리 수정
- `next 24 hours`, `horizon=24h` 같은 explicit horizon을 UNKNOWN으로 떨어뜨리던 문제 수정

수정 후 전체 회귀를 다시 통과한 상태에서 V6.1 Core를 Freeze했습니다.

## 검증 현황

| 검증 | 결과 |
|---|---:|
| inherited V3 regression | **22/22 PASS** |
| V4 improvement | **8/8 PASS** |
| V5 time-series | **7/7 PASS** |
| V6 competition | **5/5 PASS** |
| V6.1 regression | **2/2 PASS** |
| Freeze integrity | **PASS** |
| Kaggle MASTER_EVAL V1 proxy | **95.875** |
| MASTER_EVAL status | **35 PASS / 5 REVIEW / 0 FAIL** |

`95.875`는 실제 Kaggle leaderboard 점수가 아닙니다. 40개 Kaggle competition의 문제/metric/validation 성격을 기준으로 만든 **내부 proxy benchmark**입니다. 실제 Kaggle 원본 데이터 coverage는 현재 `0/40`입니다.

상세 결과:
- [`KAGGLE_MASTER_EVAL_V6_1_REPORT_KO.md`](KAGGLE_MASTER_EVAL_V6_1_REPORT_KO.md)
- [`KPI_V6_1.json`](KPI_V6_1.json)
- [`FINAL_STATUS_V6_1.json`](FINAL_STATUS_V6_1.json)
- [`V6_1_CORE_FREEZE.json`](V6_1_CORE_FREEZE.json)

## 현재 확인된 약점

V6.1은 아래 문제를 숨기지 않고 다음 버전 개선 대상으로 남겨둡니다.

1. **Time + Group 복합 Validation 부족**
   - Bike Sharing / IEEE Fraud 같은 문제에서 group-aware와 temporal validation을 함께 고려해야 함
2. **Competition 고유 metric adapter 부족**
   - M5 WRMSSE
   - COVID Week 5 weighted pinball
3. **Time-Series 일반 후보 모델 경쟁력 부족**
   - proxy 10/10에서 Persistence baseline을 이기지 못함
4. **Vision proxy 난이도 부족**
   - 현재 100점은 실제 Kaggle Vision 일반화 성능을 의미하지 않음
5. **실제 Kaggle/MLE-bench 실행 미완료**
   - 실제 competition data / leaderboard evaluation 필요

## 빠른 시작

### 1. 설치

Windows:

```cmd
setup_windows.bat
```

직접 설치:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

설치 후 현재 환경을 확인하려면:

```cmd
.venv\Scripts\python.exe verify_install.py
```

예제 CSV는 Repository에서 생성합니다.

```cmd
.venv\Scripts\python.exe examples\generate_demo_data.py
```

선택적으로 Embedding + FAISS RAG:

```cmd
setup_rag_embeddings_windows.bat
```

자세한 설치 절차: [`docs/INSTALL_KO.md`](docs/INSTALL_KO.md)

### 2. Codex 실행

Repository 루트에서:

```cmd
codex
```

예:

```text
이 CSV를 분석해줘.
Target 의미, 데이터 누수, 적절한 split, baseline, 실패 segment,
최종 운영 판단까지 검토해줘.
```

Notebook 과제:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv를 사용해.
```

Repository Skill:

```text
.agents/skills/ai-data-expert/SKILL.md
```

### 3. Harness 직접 실행

먼저 `examples\generate_demo_data.py`를 실행한 뒤:

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "수율을 예측하고 누수, split, baseline, 운영 위험을 검토" ^
  --target yield_percentage ^
  --prediction-time "before process completion" ^
  --out outputs\expert_context.json
```

Time-Series의 경우 timestamp/horizon을 명시할 수 있습니다.

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv series.csv ^
  --task "forecast the next 24 hours" ^
  --target target ^
  --modality time-series ^
  --timestamp-col timestamp ^
  --horizon 24h ^
  --out outputs\forecast_context.json
```

> `solve_timeseries_rnn_v5.py`는 현재 포함된 Steel Industry 연습 데이터 구조에 맞춘 전용 notebook solver입니다. 범용 시계열 solver로 과장하지 않습니다.

## Domain RAG

`domain_knowledge/`에 조직/공정 근거를 넣거나 `--domain-path`로 추가합니다.

권장 자료:

```text
data_dictionary.md
process_flow.md
sensor_spec.md
quality_standard.md
defect_definition.md
incident_history.md
operational_constraints.json
```

RAG의 역할은 단순 참고문 출력이 아닙니다. 구조화된 근거가 있으면 TaskSpec의 prediction time, group id, 사용 불가능 feature, business cost 판단을 바꿀 수 있습니다.

## 상태 의미

- **PASS**: 현재 evidence/verification contract에서 치명적 위반을 발견하지 못함
- **REVIEW**: 분석은 가능하지만 중요한 불확실성·가정·데이터 부족이 남음
- **FAIL**: leakage, 실행 실패, contract 위반 등으로 결과 승격 금지

`REVIEW`는 실패가 아니라, 모르는 것을 PASS로 포장하지 않기 위한 정상 상태입니다.

## 문서

- [설치 가이드](docs/INSTALL_KO.md)
- [사용 설명서](docs/USAGE_KO.md)
- [아키텍처](docs/ARCHITECTURE_KO.md)
- [개발 기록](docs/DEVELOPMENT_LOG_KO.md)
- [연구 일지](docs/RESEARCH_LOG_KO.md)
- [실사용 검증](docs/REAL_VALIDATION_KO.md)
- [V4 Hybrid RAG 보고서](V4_HYBRID_RAG_AND_USAGE_REPORT_KO.md)
- [V5 RNN Simulation 보고서](V5_RNN_SIMULATION_REPORT_KO.md)
- [V6.1 Kaggle MASTER_EVAL 보고서](KAGGLE_MASTER_EVAL_V6_1_REPORT_KO.md)

## Release 정책

- 과거 Freeze 파일과 실패 증거를 덮어쓰지 않습니다.
- V6은 회귀 실패가 발견되어 Release 기준선으로 승격하지 않았습니다.
- 현재 `main` 기준 최신 코드는 **V6.1 Candidate**입니다.
- V6.1도 아직 Production Release가 아닙니다.
- 다음 개선은 V6.1 Freeze를 보존한 채 새 버전에서 진행합니다.
