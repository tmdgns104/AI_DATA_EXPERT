# AI DATA EXPERT

**Codex에서 자동으로 사용할 수 있는 검증형 AI/Data Expert Copilot**

현재 기준 버전: **V4 Core (Frozen) + Clone-Ready Distribution**  
상태: **Pre-Production / Expert Copilot**

이 프로젝트의 목표는 단순히 모델을 자동 학습하는 것이 아닙니다.

> 문제 정의 → 데이터 의미 확인 → Domain Evidence 검색 → 전문가 선택 → 가설/실험 → 모델링 → 반박 → 검증 → PASS/REVIEW/FAIL

의 흐름으로, **그럴듯하지만 잘못된 분석을 자동으로 거부하는 데이터 분석 Agent**를 만드는 것이 목표입니다.

## 바로 설치해서 사용

### Git clone

```cmd
git clone https://github.com/tmdgns104/AI_DATA_EXPERT.git
cd AI_DATA_EXPERT
setup_windows.bat
```

또는 GitHub에서 **Code → Download ZIP** 후 압축을 풀고 해당 폴더에서:

```cmd
setup_windows.bat
```

`setup_windows.bat`은 `.venv` 생성, 의존성 설치, demo 데이터 생성 후 `verify_install.py`로 **V4 Runtime import → 실제 Expert 분류 실행 → ML Route → Verifier**까지 확인합니다. 중간 단계가 실패하면 성공 메시지를 출력하지 않습니다.

성공 기준:

```text
AI Data Expert V4 is ready.
```

그 다음 같은 폴더에서:

```cmd
codex
```

상세 설치: [`docs/INSTALL_KO.md`](docs/INSTALL_KO.md)

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

## V4 핵심

- Target missing을 새로운 Class로 취급하지 않고 **labeled / unlabeled 분리**
- `_id`, 고유 Serial, 행 순서 Proxy 등 **식별자/순서 누수 자동 경고 및 제외**
- 반복 Entity 또는 `Shot/Cycle/Sequence` reset을 이용한 **Group/Run split 후보 추론**
- Test set을 모델/Threshold 선택에 사용하지 않는 **최종 Holdout 격리**
- BM25 + Vector + Metadata + Structured Facts 기반 **Hybrid Domain RAG**
- RAG 결과를 prediction time / group / unavailable feature / business cost에 반영
- Notebook 실행뿐 아니라 Target missing, ID, Group overlap, Test isolation을 검사하는 **Semantic Validator**
- Windows UTF-8 / fail-fast 설치 / Repository-local Jupyter 설정
- PyTorch Tabular DL / Pixel CNN 실제 실행 경로
- 희귀 클래스 표본이 너무 적으면 성능 숫자를 과신하지 않고 **REVIEW 유지**
- PyTorch checkpoint는 `state_dict` 중심 안전 저장/재로딩 원칙 사용
- Tacit Expert Heuristics + Source Trace

## 검증 현황

| 검증 | 결과 |
|---|---:|
| V4 improvement tests | **8/8 PASS** |
| inherited V3 regression | **22/22 기능 PASS** |
| V4 targeted sealed | **96/96 PASS** |
| Clone-ready install smoke | **PASS** |
| V4 Notebook E2E isolated rerun | **PASS** |

> `96/96`은 V4에서 발견한 결함을 겨냥한 targeted regression/generalization 시험입니다. 실무 전체 정확도 100%를 의미하지 않습니다.

### 실제 Codex 사용 검증

2026-08-26 실제 사용으로 추가 검증했습니다.

- **극단적 불균형 DNN 분류**: 3,000행 중 불량 9건. Accuracy를 성공 기준으로 삼지 않고 Macro-F1, Balanced Accuracy, defect Recall, PR-AUC를 사용. 운영 판단 `REVIEW`.
- **3-class Vision CNN**: 실제 이미지 픽셀 학습, Train/Validation/Test 중복 검사, best checkpoint 복원, 저장 모델 안전 재로딩 검증. Test Accuracy **0.9444**, Macro-F1 **0.9441**.

상세: [`docs/REAL_VALIDATION_KO.md`](docs/REAL_VALIDATION_KO.md)

## 사용 예

설치가 끝나면 demo CSV가 `examples/`에 자동 생성됩니다.

Codex에서:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv를 사용해.
```

Codex 없이 demo:

```cmd
run_demo_without_codex.bat
```

Harness 직접 실행:

```cmd
.venv\Scripts\python.exe .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "estimate yield_percentage from process variables" ^
  --target yield_percentage ^
  --prediction-time "before process completion" ^
  --out outputs\expert_context.json
```

Repository Skill: `.agents/skills/ai-data-expert/SKILL.md`

## Hybrid RAG

기본 설치에서는 BM25 + offline vector fallback을 사용할 수 있습니다. Embedding + FAISS를 추가하려면:

```cmd
setup_rag_embeddings_windows.bat
```

회사/프로젝트 문서는 `domain_knowledge/`에 넣거나 `--domain-path`로 지정합니다.

## 상태 의미

- **PASS**: 현재 검증 계약에서 치명적 문제 없음
- **REVIEW**: 분석 가능하지만 예측 시점, 비용, 라벨 의미, Group 정의, 표본 수 등 중요한 불확실성이 남음
- **FAIL**: Expert 실행 실패 또는 leakage/validation 계약 위반으로 결과 승격 금지

## 문서

- [설치 가이드](docs/INSTALL_KO.md)
- [사용 설명서](docs/USAGE_KO.md)
- [아키텍처](docs/ARCHITECTURE_KO.md)
- [연구 일지](docs/RESEARCH_LOG_KO.md)
- [개발 기록](docs/DEVELOPMENT_LOG_KO.md)
- [실사용 검증 기록](docs/REAL_VALIDATION_KO.md)
- [V4 Hybrid RAG / 실사용 개선 보고서](V4_HYBRID_RAG_AND_USAGE_REPORT_KO.md)
- [현재 테스트 상태](TEST_STATUS.json)
- [V4 Sealed 결과](evaluation/sealed/SEALED_RESULTS_V4.json)

## 현재 한계

- `sentence-transformers + FAISS` 경로는 구현되어 있으나 기존 V4 Sandbox 평가는 **BM25 + char-TFIDF fallback** 기준
- 실제 Domain RAG 품질은 Data Dictionary / Sensor Spec / Process Flow / Quality Spec 품질에 의존
- Group/Run 추론은 휴리스틱일 수 있으므로 MES/Batch 실제 lineage가 있으면 그것을 우선해야 함
- Vision / Tabular / Time-Series의 의미 검증 계약은 다음 버전에서 modality별로 더 분리할 필요가 있음
- Expert 간 공용 Evidence State가 더 필요함
- 완전 자율 Production 승인 도구가 아니라 **Human Review가 가능한 Expert Copilot** 단계

## 다음 개선 전 원칙

V4 핵심 분석 Runtime과 평가 결과는 기준선으로 보존합니다. 설치/배포 hardening은 별도 release 기록으로 남기며, 다음 기능 개선은 V5에서 진행합니다.
