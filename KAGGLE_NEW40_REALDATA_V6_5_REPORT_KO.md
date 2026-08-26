# AI Data Expert V6.5 — 새로운 Kaggle 40문제 직접 실행 평가

## 목적

기존 `catalog_v1`의 40문제를 재사용하지 않고, 회귀/분류/시계열/Vision 각 10개씩 **새로운 Kaggle Competition 문제 명세 40개**를 별도로 구성했습니다.

Kaggle 자격증명이 없는 실행 환경이므로 원본 Competition train/test 파일을 내려받아 leaderboard에 제출한 평가는 아닙니다. 대신 각 Competition의 modality/target/validation 성격에 맞는 **오프라인 실제 데이터셋**으로 전처리, baseline, 후보 모델 학습, Validation 선택, Test 평가까지 실제 코드로 실행했습니다.

## 결과

| 유형 | 실행 완료 | 선택 모델이 baseline 상회 |
|---|---:|---:|
| 회귀 | 10/10 | 6/10 |
| 분류 | 10/10 | 9/10 |
| 시계열 | 10/10 | 9/10 |
| Vision | 10/10 | 10/10 |
| **합계** | **40/40** | **34/40** |

- 실행 실패: **0/40**
- 원본 Kaggle 데이터 coverage: **0/40**
- Leaderboard score: **측정하지 않음**

6개 문제에서는 단순 baseline이 선택 모델보다 강하거나 동등했습니다. 이 경우 baseline을 실패로 숨기지 않고 그대로 결과에 남겼습니다.

시계열에서는 10개 중 9개가 persistence 계열 baseline을 이겼고, 한 문제에서는 persistence가 champion으로 남았습니다. 이는 V6.5의 baseline-inclusive selection 규칙이 실제 실행에서 동작했다는 증거입니다.

## 해석

이 평가는 “40개 Kaggle 대회를 정복했다”는 의미가 아닙니다. **서로 다른 40개 Competition 명세를 보고 문제 유형에 맞는 데이터 처리/검증/모델 비교를 끝까지 실행할 수 있는지**를 보는 내부 일반화 평가입니다.

실제 Kaggle 성능을 평가하려면 Kaggle API credential과 Competition별 원본 데이터/규칙 동의가 필요합니다.
