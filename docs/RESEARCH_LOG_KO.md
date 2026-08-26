# AI DATA EXPERT 연구일지

작성 기준일: **2026-08-26**

이 문서는 점수만 나열하지 않고, 각 버전에서 무엇을 발견했고 왜 다음 버전이 필요했는지를 기록합니다.

## S0 — Agent 없는 순정 분석

제조 수율 예측 Notebook을 직접 분석하는 형태에서 시작.

- 정형 데이터 모델 비교 가능
- DNN이 작은 Tabular 데이터에서 자동으로 우월하지 않음을 확인
- Router / Expert / Verifier / Holdout / Freeze / Evidence 구조가 없었음

핵심 질문:

> 하나의 Notebook을 잘 푸는 것과, 다양한 문제에서 잘못된 분석을 막는 Expert System은 다른 문제다.

## Data Expert V2/V3

초기 Benchmark에서 반복 Entity leakage와 ID trap 발견.

개선:

- Entity/group leakage guard
- pure ID/identifier trap
- baseline / holdout discipline

Known benchmark에서는 높은 점수를 얻었지만 새로운 Holdout에서 Target transform, calibration, dedup-before-split 등 새 blind spot이 드러남.

교훈:

> 이미 본 시험의 100%보다 새 Sealed Holdout 실패가 더 중요하다.

## Multi-Expert Supervisor

Data Analyst / ML / DL / Vision / Time-Series / Big Data / MLOps / Verifier로 역할을 분리.

- Agent = 도메인 책임
- Skill = 특정 방법
- 모델 하나당 Agent를 만들지 않음

새 Holdout에서 `do not train`, audit-only, monitoring existing model 같은 Intent 문제를 발견.

## Expert Reasoning V1

공통 reasoning contract 도입.

```text
UNDERSTAND
INSPECT
QUESTION
HYPOTHESIZE
TEST
COMPARE
DECIDE
CHALLENGE
RISK
CONFIDENCE
```

멀티라벨 Vision, censored survival에서 실패하며 도메인별 reasoning 차이를 확인.

## Tacit Expert V2

공개적으로 검증 가능한 전문가 원칙을 구조화.

예:

- Tukey: EDA / structure scan
- Breiman: honest out-of-sample prediction comparison
- Gelman: iterative workflow / model checking
- Zinkevich: simple first model / reliable pipeline
- Karpathy: data first / small-batch sanity / silent failure
- Hyndman: naive baseline / chronological validation
- Harrell: validation / censor-aware survival

중요 원칙:

> 개인의 비공개 암묵지를 주장하지 않고, 공개 Source에서 반복적으로 확인되는 원칙만 Source Trace와 함께 사용한다.

## Codex Skill V1 — 실제 Codex 자동 사용

Codex Repository Skill 형태로 패키징.

실제 DNN Notebook 과제를 실행하며 첫 실사용 결함 발견:

- classification path가 regression 코드와 섞임
- Notebook Solver가 Test 결과로 모델 선택
- 영어 Router 취약
- Runtime Verifier 미연결

제품 평가 기준선을 약 68/100으로 설정.

## Codex Skill V2

개선:

- Regression / Classification 완전 분리
- Train / Validation / Final Test discipline
- 영어/부정문 Router
- Runtime Verifier
- Expert failure isolation

새 Sealed에서 keyword Router의 의미 한계를 확인.

## Codex Skill V3

추가:

- TaskSpec
- Intent-first Router
- Domain RAG
- Hypothesis / Experiment
- Challenger
- actual PyTorch DL
- actual pixel CNN

Frozen Sealed에서 높은 점수를 얻었지만 두 실패를 보존:

- no-training intent와 Verifier 해석 불일치
- 실제 3-class Pixel CNN 성능 부족

## V3 실제 제조 분류 사용 — 가장 중요한 발견

실제 다이캐스팅 형태 데이터에서:

- `Machine_Status` Target missing을 Class로 잘못 볼 수 있음
- `_id`가 행 순서 Proxy 역할
- Shot reset으로 여러 생산 run 후보가 보임
- Random holdout macro-F1 약 0.907
- Run-isolated Test macro-F1 약 0.771
- status 1 recall 약 0.381

핵심 교훈:

> 더 높은 점수를 만드는 것이 아니라, 잘못된 높은 점수를 자동으로 거부해야 한다.

## Codex Skill V4 — Data Meaning + Hybrid Domain RAG

개선:

- Target missing → labeled/unlabeled 분리
- ID/row-order proxy detector
- Entity/Sequence reset 기반 Group split 후보
- Semantic Notebook Validator
- Windows 실패 전파 / UTF-8
- Hybrid Domain RAG
- Structured fact → TaskSpec/Feature/Split 반영

Targeted Sealed: 96/96 PASS.

단, 이 점수는 실무 정확도 100%가 아니라 해당 결함군에 대한 regression/generalization evidence.

## V4 실제 DNN 불균형 분류

3,000행 중 defect 9건(0.3%).

분석 원칙:

- Accuracy 단독 사용 금지
- stratified split
- balanced logistic baseline
- weighted-loss PyTorch DNN
- small-batch overfit sanity
- Validation threshold
- Final Test 1회

최종 상태:

- 실행/의미 검증 PASS
- Macro-F1 약 0.5246
- Balanced Accuracy 약 0.7291
- defect recall 0.5
- 운영 적용: REVIEW

교훈:

> 매우 작은 양성 표본에서는 단일 Threshold와 지표를 과신하면 안 된다.

## V4 실제 Vision 3-class

실제 이미지 폴더 기반 CNN 작업.

결과:

- Test Accuracy 0.9444
- Balanced Accuracy 0.9444
- Macro-F1 0.9441
- 72장 중 4장 오분류
- Train/Validation/Test 완전 중복 이미지 0
- 체크포인트 저장 및 안전 재로딩 확인

새로 발견한 시스템 문제:

- Vision Expert와 DL Expert 상태가 완전히 공유되지 않음
- Tabular 중심 Semantic Validator가 Vision 구조를 완전히 이해하지 못함
- torchvision / PyYAML dependency 정리 필요
- Windows/Jupyter 경로와 UTF-8 마찰

## 현재 연구 결론

현재 가장 큰 성숙도 향상은 모델 수 증가가 아니라 다음에서 나왔다.

1. 잘못된 Split 거부
2. Test 오염 거부
3. ID/Proxy leakage 거부
4. Target semantics 확인
5. 희귀 Class 실패 노출
6. 결과가 낮아도 REVIEW/FAIL을 유지
7. Domain Evidence를 실제 분석 행동에 연결

다음 버전은 현재 V4를 보존한 뒤 별도 V5로 연구한다.
