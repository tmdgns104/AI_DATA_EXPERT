# AI DATA EXPERT V6.1 Candidate Architecture

## Runtime Flow

```text
User / CSV / Notebook
        ↓
TaskSpec V6.1
        ↓
Data Guard V4
        ↓
Hybrid Domain RAG V5
        ↓
Intent / Modality Routing
        ↓
Expert Execution
  ├─ Data Analyst
  ├─ Machine Learning
  ├─ Deep Learning
  ├─ Vision AI
  ├─ Time-Series
  ├─ Big Data
  └─ MLOps
        ↓
Shared Evidence Store
        ↓
Argument Ledger
        ↓
Hypothesis / Experiment Evidence
        ↓
Local Challenger
        ↓
Modality Verifier V6.1
        ↓
PASS / REVIEW / FAIL
```

Competition-aware task는 별도 계약을 추가합니다.

```text
Competition task
      ↓
CompetitionSpec V6
      ↓
Competition Planner
  ├─ metric / direction
  ├─ validation contract
  ├─ leakage risk
  ├─ submission mode
  └─ metric runtime exactness
      ↓
Shared Evidence + Argument Ledger
      ↓
Competition Verifier
      ↓
Human-friendly Renderer
```

## 현재 활성 핵심 모듈

| 파일 | 역할 |
|---|---|
| `data_expert/enhanced_system.py` | V6.1 진입점 |
| `data_expert/v6_1_system.py` | V6.1 orchestration override |
| `data_expert/task_spec_v61.py` | forecast negation / explicit horizon 회귀 수정 |
| `data_expert/v6_system.py` | Competition-aware layer |
| `data_expert/competition_spec_v6.py` | competition contract |
| `data_expert/competition_planner_v6.py` | validation/metric/submission planning |
| `data_expert/competition_verifier_v6.py` | competition contract verification |
| `data_expert/output_renderer_v6.py` | human-friendly output contract |
| `data_expert/v5_system.py` | Shared Evidence / Argument Ledger / TS specialist layer |
| `data_expert/shared_evidence_v5.py` | 공용 evidence state |
| `data_expert/argument_ledger_v5.py` | 논증 상태 추적 |
| `data_expert/time_series_dl_v5.py` | TS specialist execution |
| `data_expert/domain_rag_v5.py` | V5 domain retrieval layer |
| `data_expert/data_guard_v4.py` | Target/ID/Group/Time guard |

## 설계 원칙

1. 문제 정의가 모델보다 먼저입니다.
2. Target missing은 Class가 아닙니다.
3. ID/row-order proxy는 기본 Feature에서 제외합니다.
4. Group/Time 구조는 데이터 생성 과정을 기준으로 검증 전략을 선택합니다.
5. Final Test는 선택에 사용하지 않습니다.
6. Baseline을 이기지 못한 복잡한 모델은 성공으로 포장하지 않습니다.
7. Domain RAG는 근거이며, 출처가 없으면 사실을 만들어내지 않습니다.
8. Argument Ledger의 `SUPPORTED`는 현재 evidence에서 살아남았다는 뜻이지 영구적 진실이 아닙니다.
9. Competition metric은 generic metric으로 대체하지 않습니다.
10. exact metric reproduction이 불가능하면 `REVIEW`를 허용합니다.

## 현재 알려진 구조적 한계

- Time + Group 복합 validation planner가 아직 없음
- M5 WRMSSE, weighted pinball 등 일부 competition 고유 metric adapter가 proxy 수준
- general-purpose time-series candidate가 강한 persistence baseline을 안정적으로 이기지 못함
- Vision proxy benchmark가 실제 Kaggle보다 쉬움
- 실제 Kaggle/MLE-bench 원본 데이터 평가는 미수행
- embedding + FAISS 경로는 환경별 runtime verification이 필요함
