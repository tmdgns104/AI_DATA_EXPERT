# AI DATA EXPERT

**Codex에서 자동으로 사용할 수 있는 검증형 AI/Data Expert Copilot**

현재 기준 버전: **V4 (Frozen)**  
상태: **Pre-Production / Expert Copilot**

이 프로젝트의 목표는 단순히 모델을 자동 학습하는 것이 아닙니다.

> 문제 정의 → 데이터 의미 확인 → Domain Evidence 검색 → 전문가 선택 → 가설/실험 → 모델링 → 반박 → 검증 → PASS/REVIEW/FAIL

의 흐름으로, **그럴듯하지만 잘못된 분석을 자동으로 거부하는 데이터 분석 Agent**를 만드는 것이 목표입니다.

## 현재 핵심 구조

```text
User / CSV / Notebook
        ↓
TaskSpec
        ↓
Intent Router
        ↓
Hybrid Domain RAG
  ├─ BM25
  ├─ Vector similarity
  ├─ Metadata boost
  └─ Structured facts
        ↓
Data / ML / DL / Vision / TS / BigData / MLOps Experts
        ↓
Hypothesis + Experiment
        ↓
Challenger
        ↓
Runtime Verifier
        ↓
PASS / REVIEW / FAIL
```

## V4에서 달라진 점

- Target missing을 새로운 Class로 취급하지 않고 **labeled / unlabeled 분리**
- `_id`, 고유 Serial, 행 순서 Proxy 등 **식별자/순서 누수 자동 경고 및 제외**
- 반복 Entity 또는 `Shot/Cycle/Sequence` reset을 이용한 **Group/Run split 후보 추론**
- Test set을 모델/Threshold 선택에 사용하지 않는 **최종 Holdout 격리**
- BM25 + Vector + Metadata + Structured Facts 기반 **Hybrid Domain RAG**
- RAG 결과를 단순 표시하지 않고 prediction time / group / unavailable feature / business cost에 반영
- Notebook 실행 성공뿐 아니라 Target missing, ID, Group overlap, Test isolation을 검사하는 **Semantic Validator**
- Windows UTF-8 및 설치 실패 종료 코드 개선
- PyTorch Tabular DL / Pixel CNN 실제 실행 경로
- Tacit Expert Heuristics + Source Trace

## 검증 현황

| 검증 | 결과 |
|---|---:|
| V4 improvement tests | **8/8 PASS** |
| inherited V3 regression | **22/22 기능 PASS** |
| V4 targeted sealed | **96/96 PASS** |
| Freeze integrity | **PASS** |

> `96/96`은 V4에서 발견한 결함을 겨냥한 targeted regression/generalization 시험입니다. 실무 전체 정확도 100%를 의미하지 않습니다.

### 실제 사용 검증

2026-08-26 Codex 실사용으로 두 종류의 과제를 추가 검증했습니다.

- **극단적 불균형 DNN 분류**: 3,000행 중 불량 9건. Accuracy를 성공 기준으로 삼지 않고 Macro-F1, Balanced Accuracy, defect Recall, PR-AUC를 사용. 최종 운영 판단은 `REVIEW`.
- **3-class Vision CNN**: 실제 이미지 픽셀 학습, Train/Validation/Test 중복 검사, best checkpoint 복원, 저장 모델 안전 재로딩까지 검증. Test Accuracy 0.9444 / Macro-F1 0.9441.

상세 기록: [`docs/REAL_VALIDATION_KO.md`](docs/REAL_VALIDATION_KO.md)

## 빠른 시작

### 1. 설치

```cmd
setup_windows.bat
```

선택적으로 Embedding + FAISS RAG를 쓰려면:

```cmd
setup_rag_embeddings_windows.bat
```

### 2. Codex 실행

```cmd
codex
```

예:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv를 사용해.
```

Repository Skill은 `.agents/skills/ai-data-expert/SKILL.md`에 있습니다.

### 3. Harness 직접 실행

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "estimate yield_percentage from process variables" ^
  --target yield_percentage ^
  --prediction-time "before process completion" ^
  --out outputs\expert_context.json
```

## 상태 의미

- **PASS**: 현재 검증 계약에서 치명적 문제 없음
- **REVIEW**: 분석은 가능하지만 예측 시점, 비용, 라벨 의미, Group 정의, 데이터량 등 중요한 불확실성이 남음
- **FAIL**: Expert 실행 실패 또는 leakage/validation 계약 위반으로 결과 승격 금지

## 문서

- [사용 설명서](docs/USAGE_KO.md)
- [아키텍처와 현재 구조](docs/ARCHITECTURE_KO.md)
- [연구 일지](docs/RESEARCH_LOG_KO.md)
- [개발 기록](docs/DEVELOPMENT_LOG_KO.md)
- [실사용 검증 기록](docs/REAL_VALIDATION_KO.md)
- [V4 Hybrid RAG / 실사용 개선 보고서](V4_HYBRID_RAG_AND_USAGE_REPORT_KO.md)

## 현재 한계

- `sentence-transformers + FAISS` 경로는 구현되어 있으나 V4 샌드박스에서는 의존성/로컬 임베딩 모델 부재로 **BM25 + char-TFIDF fallback**으로 검증됨
- 실제 Domain RAG 품질은 조직의 Data Dictionary / Sensor Spec / Process Flow / Quality Spec 품질에 의존
- Group/Run 추론은 휴리스틱일 수 있으므로 MES/Batch 실제 lineage가 있으면 그것을 우선해야 함
- Vision / Tabular / Time-Series의 의미 검증 계약이 아직 완전히 modality별로 분리되어 있지 않음
- Expert 간 공용 Evidence State가 더 필요함
- 완전 자율 Production 승인 도구가 아니라 **Human Review가 가능한 Expert Copilot** 단계

## 다음 개선 전 원칙

V4는 현재 기준점으로 동결합니다. 다음 개선은 V5에서 별도로 진행하고, 기존 V4 결과와 실패를 보존합니다.
