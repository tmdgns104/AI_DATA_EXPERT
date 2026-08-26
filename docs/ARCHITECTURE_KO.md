# AI DATA EXPERT V4 Architecture

## Runtime Flow

```text
User / File / Notebook
        ↓
TaskSpec V4
        ↓
Data Guard V4
        ↓
Intent / Modality Routing
        ↓
Hybrid Domain RAG V4
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
Hypothesis Engine
        ↓
Experiment Evidence
        ↓
Challenger
        ↓
Runtime Verifier
        ↓
PASS / REVIEW / FAIL
```

## 핵심 모듈

- `data_expert/v4_system.py`: V4 상위 실행 경로
- `data_expert/task_spec_v4.py`: 문제/Target/예측시점/분할/비용 계약
- `data_expert/data_guard_v4.py`: Target missing, ID/row proxy, Group/Sequence guard
- `data_expert/domain_rag_v4.py`: Hybrid Domain RAG
- `data_expert/advanced_ml_v4.py`: V4 ML diagnostics
- `data_expert/dl_engine_v4.py`: DL/Vision 실행 보강
- `data_expert/hypothesis_engine_v4.py`: 가설/검증 구조
- `data_expert/challenger_v4.py`: 반박 및 운영 위험 검토

## RAG V4

```text
Domain documents
   ↓
Chunk + Metadata
   ↓
BM25 lexical
   +
Vector similarity
   +
Metadata boost
   +
Structured facts
   ↓
Retrieved Evidence
   ↓
TaskSpec enrichment + Expert context injection
```

Semantic dependency가 없으면 char-TFIDF fallback을 사용합니다.

## 데이터 안전 원칙

1. Target missing은 Class가 아님
2. 고유 ID/행 순서 Proxy는 기본 Feature에서 제외
3. Group/Time 구조가 있으면 Random split을 기본값으로 사용하지 않음
4. Final Test는 모델/Threshold 선택에 사용하지 않음
5. Feature importance는 causality가 아님
6. Drift 하나만 보고 자동 재학습하지 않음
7. Domain 근거가 없으면 운영 사실을 만들어내지 않음

## 아직 V5 후보인 구조

아래는 아직 V4에 완전히 구현된 기능이 아니라 다음 개선 후보입니다.

- Argument Ledger / Argument Graph
- 단계별 Local Challenger
- Shared Evidence Store
- Modality-specific Semantic Verifier
- Expert 간 상태 완전 공유
- Threshold uncertainty / cost-aware decision engine 강화
