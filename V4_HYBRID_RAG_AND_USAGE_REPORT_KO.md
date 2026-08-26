# AI Data Expert Codex Skill V4 — Hybrid RAG + 실사용 결함 개선 보고서

## 결론

V4는 V3 실사용 로그에서 드러난 데이터 의미 결함과 Domain RAG의 약점을 동시에 개선했다.

- V4 개선 테스트: **8/8 PASS**
- V3 기존 기능 회귀: **22/22 기능 PASS**
- Freeze 후 Targeted Sealed: **100.00% (96/96)**
- Frozen 핵심 파일 무결성: **PASS**

이 Sealed 100%는 이번에 발견한 결함을 겨냥한 Targeted regression/generalization 시험이다. 실무 전체 정확도 100%라는 뜻이 아니다.

## 1. RAG V4

```text
TaskSpec
  ↓
Hybrid Domain RAG
  ├─ BM25 lexical
  ├─ vector similarity
  ├─ metadata boost
  └─ structured JSON facts
  ↓
Retrieved source text + source-bounded facts
  ↓
TaskSpec enrichment
  ├─ prediction_time
  ├─ business_cost
  ├─ group_id
  ├─ unavailable features
  └─ observation_unit
  ↓
Expert Context Injection
  ↓
Hypothesis / Challenger / Verifier
```

### Retrieval

기본 결합:
- BM25 45%
- vector similarity 50%
- metadata boost 최대 20%

`sentence-transformers`와 로컬 embedding model이 있고 `faiss-cpu`가 설치되어 있으면 실제 embedding + FAISS inner-product 검색을 사용한다. 이 환경에는 두 의존성과 로컬 모델이 없어 Sealed 실행은 **BM25 + char-TFIDF vector fallback**으로 수행했다.

### Structured Fact Injection

JSON 문서의 `facts`는 검색된 Source에 한정해 적용한다. 지원 예:
- prediction_time
- business_cost / cost_matrix
- feature_unavailable
- group_id / entity_group
- observation_unit

Target이 다른 Fact는 적용하지 않아 cross-task pollution을 막는다.

## 2. 실사용 로그 반영

### Target missing

V3에서 `Machine_Status` 결측이 `nan`이라는 3번째 Class처럼 처리될 수 있었다. V4는 labeled/unlabeled를 분리하고 학습/평가는 labeled row만 사용하며, unlabeled row는 선택 모델로 별도 prediction한다.

### ID / row-order proxy

`_id`, unique serial, row index, near-monotonic sequence proxy를 감지해 feature에서 제외한다.

### 생산 Run / Entity split

반복 `equipment_id`, `lot`, `patient`, `product` 계열 Entity를 우선 Group 후보로 사용한다. 명시 Entity가 없고 `Shot/Cycle/Sequence`가 reset되면 reset 지점을 생산 run boundary 후보로 만들어 Group Split한다.

### Honest score

이 개선의 목적은 점수를 올리는 것이 아니라 Random Split의 과대평가를 막는 것이다. V3 실사용에서 random macro-F1 약 0.907이 run-isolated test 0.771로 내려간 사례가 있었기 때문에 V4는 split semantics를 모델 선택보다 앞에 둔다.

### Semantic Notebook Validator

Notebook이 실행되는지만 확인하지 않고 다음을 검사한다.
- target missing이 class로 들어가지 않았는가
- ID proxy가 제외됐는가
- Group-aware split이 구현됐는가
- validation/test가 분리됐는가
- threshold가 validation에서 선택됐는가
- class-wise report가 있는가
- 근거 없이 production-ready라고 주장하는가

### Windows

`setup_windows.bat`은 UTF-8 모드를 켜고 pip/install 단계 실패 시 즉시 non-zero exit code로 종료한다.

## 3. Targeted Sealed 결과

총 weighted checks: **96**  
PASS: **96**  
FAIL: **0**

시험 범위:
- partial-label classification
- `_id` order proxy
- Shot reset production run
- repeated equipment entity split
- chronological split
- RAG prediction-time fact injection
- RAG unavailable-feature exclusion
- target-scoped fact pollution guard
- no-train Vision negation
- actual 3-class pixel CNN
- Notebook semantic E2E
- custom operational cost/group RAG facts
- Windows setup contract
- expert failure isolation

## 4. 이번 결과에서 새로 파악한 개선점

### P0 — 실제 Embedding + FAISS runtime 검증
코드는 준비됐지만 sandbox에서는 실행되지 않았다. Windows V4 환경에서 `setup_rag_embeddings_windows.bat`로 설치하고 로컬 multilingual embedding model을 지정한 뒤 lexical-only/fallback/hybrid retrieval 비교 benchmark가 필요하다.

### P0 — Domain RAG 평가세트
현재는 synthetic/프로젝트 문서 중심이다. 실제 Data Dictionary, Sensor Spec, Process Flow, Quality Spec, incident 문서에서 Recall@K / source precision / wrong-fact injection을 측정해야 한다.

### P1 — Group inference provenance
Shot reset은 강한 힌트지만 진짜 생산 run인지 보장하지 않는다. MES batch/run ID가 있으면 그것을 우선하고, heuristic-derived group은 REVIEW Evidence로 표시해야 한다.

### P1 — Partial labels의 missing mechanism
라벨 결측을 분리하는 것만으로 끝나지 않는다. missing-at-random인지, 특정 설비/상태에서 라벨이 누락되는지 분석해야 selection bias를 줄일 수 있다.

### P1 — ID/Proxy detector 고도화
고유 ID뿐 아니라 target encoding된 code, future inspection field, monotonic timestamp proxy, duplicated entity snapshot 같은 semantic leakage를 더 탐지해야 한다.

### P1 — Cost-aware classification
business cost를 RAG에서 얻으면 threshold optimization이 실제 expected cost를 최소화하도록 Solver/ML Expert에 완전히 연결해야 한다.

### P2 — Windows E2E
한글 경로/CP949/WMIC/Jupyter warning을 포함한 실제 Windows CI가 필요하다. 현재는 script contract 수준 검증이다.

## 5. 판단

V4의 가장 큰 변화는 RAG를 '검색 결과 표시'에서 **분석 행동을 바꾸는 Domain Evidence**로 만든 것이다. 검색된 사실이 prediction time, feature exclusion, group split, business cost에 직접 반영되며 Expert에게 원문 Evidence도 전달된다.

현재 위치는 **실무 데이터 의미를 더 정직하게 다루는 Pre-Production Data/ML Copilot**에 가깝다. 다음 성숙도 점프는 모델 추가보다 실제 기업 문서 RAG 평가와 real shadow dataset 검증에서 나올 가능성이 크다.
