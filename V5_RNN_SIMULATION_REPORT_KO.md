# AI Data Expert V5 Candidate — RNN/LSTM Targeted Simulation

## 결론

업로드된 `순환신경망 연습_19_260826_Question.ipynb`를 V5 개선 검증 문제로 사용했다.
이 문제는 V5 개발 중 직접 사용되었으므로 **Blind/Sealed 평가가 아니라 Targeted Integration Simulation**이다.

### 개발/회귀 검증

- V5 Time-Series 전용 테스트: **7/7 PASS**
- V4 개선 회귀 호환성: **8/8 PASS** (8번 Notebook E2E는 전체 suite timeout 후 단독 실행 PASS)
- 생성 답안 Notebook: **22 cells / execution errors 0**
- Runtime Verifier: **25/26 = 96.15% / REVIEW**
- Hard Error: **0**
- 남은 Warning: **forecast horizon 미명시 1건**

## V4에서 이 문제를 봤을 때의 구조적 문제

V4 TaskSpec/Router만 적용하면:

- problem type: `regression`
- timestamp: `None`
- primary intent: `DL_TRAIN`
- route: `data-analyst -> deep-learning`
- `FORECAST` intent 미선택

즉 RNN/LSTM이라는 모델 이름은 인식했지만 **시계열 예측 문제로 완전히 구조화하지 못했다.**

## V5 개선

### 1. Time-Series TaskSpec

- problem type을 `forecasting`으로 명시
- `date`를 timestamp로 탐지
- `TRAIN_MODEL + COMPARE_MODELS + FORECAST + DL_TRAIN`을 함께 구조화
- split을 `chronological train/validation/test`로 고정
- primary metric: RMSE, secondary: MAE/R2
- 과제에서 horizon이 없으므로 `one-step ahead`를 **가정으로 기록하고 REVIEW 유지**

### 2. Shared Evidence Store

Time-Series Expert가 만든 다음 Evidence를 Supervisor/Verifier가 같은 상태로 공유한다.

- TaskSpec / routing
- timestamp integrity
- model decision
- expert markers
- RAG backend

현재 record count: **5**

### 3. Argument Ledger

네 개의 논증 노드를 실제로 생성했다.

- `H-TIME-001`: 원본 timestamp를 그대로 쓸 수 있는가?
- `H-SPLIT-001`: random split이 허용되는가?
- `H-MODEL-001`: SimpleRNN/LSTM/baseline 중 명확한 우승자가 있는가?
- `H-HORIZON-001`: 예측 horizon이 무엇인가?

상태: Supported 3 / Open·Inconclusive 1

### 4. Modality-specific Time-Series Verifier

Tabular 규칙 대신 다음을 검사한다.

- timestamp integrity
- chronological split
- train-only scaling
- naive baseline
- actual RNN execution
- RNN/LSTM comparison
- final Test isolation
- Argument Ledger
- Shared Evidence Store
- forecast horizon confirmation

### 5. RAG Wrong-Domain Guard

V4 RAG는 이 문제에서도 `manufacturing_example.md`를 검색 후보로 가져왔다.
V5는 Target/도메인 직접 근거가 없는 demo 문서를 Evidence로 주입하지 않는다.

결과:

- RAG status: **NO_MATCH**
- rejected: **2개**
- `manufacturing_example.md`: `target_mismatch_demo_evidence`로 거절
- Generic Domain README: Evidence가 아니므로 거절

**NO_MATCH가 wrong-fact injection보다 안전하다는 원칙을 적용했다.**

## 데이터에서 실제로 발견한 문제

원본 `date`에는 매일 마지막 자정이 동일 날짜의 `00:00`으로 기록되어 그대로 파싱하면 시간축이 뒤로 간다.
`NSM`이 23시대에서 0으로 reset되는 행을 다음 날짜 자정으로 보정했다.

- timestamp parse failure: 0
- repaired midnight rows: **365**
- duplicate timestamps after repair: 0
- monotonic after repair: True
- dominant interval: **15분**
- irregular interval after repair: 0

## 실제 모델 시뮬레이션

입력은 이전 32개 `Usage_kWh` 관측(8시간), target은 다음 15분 `Usage_kWh`로 두었다.
미래 시점 이용 가능성이 불명확한 다른 공정 컬럼은 넣지 않았다.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| LastValueBaseline | 5.0145 | 11.6146 | 0.8612 |
| SimpleRNN | 5.6399 | 10.6983 | 0.8822 |
| **LSTM** | **4.9892** | **10.5320** | **0.8858** |

Validation RMSE 기준 선택: **LSTM**

이 실행에서는 LSTM이 Validation과 Test RMSE 모두 가장 좋았고, baseline보다도 Test MAE/RMSE가 소폭 개선됐다.
다만 한 번의 chronological holdout과 한 seed만으로 운영 우위를 확정할 수는 없다.

## 최종 판단

**REVIEW**

이유는 모델 실패가 아니라 과제에 forecast horizon이 명시되지 않았기 때문이다.
실습용으로는 15분 one-step forecast가 합리적이지만, 실제 요구사항에서는 horizon을 먼저 확정해야 한다.

## 다음 후보

1. rolling-origin backtest
2. multi-seed RNN/LSTM 안정성
3. forecast horizon 1/4/16/96 step 비교
4. 미래 이용 가능한 외생변수만 별도로 정의한 multivariate RNN/LSTM
5. 실제 Embedding+FAISS RAG benchmark
