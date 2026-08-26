# AI DATA EXPERT V6.1 - Kaggle MASTER_EVAL 40 결과

기준일: 2026-08-26

## 결론

- Strict Composite: **3835 / 4000 = 95.875**
- PASS / REVIEW / FAIL: **35 / 5 / 0**
- Critical Error: **0건**
- Proxy model이 baseline을 이긴 문제: **30/40 (75.0%)**
- 실제 Kaggle 원본 데이터/리더보드 커버리지: **0/40**

> 이 점수는 Kaggle 리더보드 점수가 아님. 실제 Kaggle 대회 40개의 문제/metric/validation 성격을 사용하고 deterministic proxy data로 Agent의 분석 계약을 시험한 내부 Benchmark임.

## 왜 V6이 아니라 V6.1인가

- 첫 V6 Freeze 후 기존 22개 회귀에서 `do not forecast`, 명시적 `24h horizon` 2개가 깨진 것을 발견했음.
- 점수가 높아도 기존 기능이 깨졌으면 Release하지 않는 원칙을 적용해 V6은 Reject 처리함.
- V6.1에서 두 회귀만 수정한 뒤 전체 기존 회귀를 통과시키고 다시 Freeze한 후 MASTER_EVAL을 실행함.

## 카테고리별 결과

| 종류 | 점수 | KPI | PASS | REVIEW | FAIL |
|---|---:|---:|---:|---:|---:|
| regression | 985/1000 | 98.5 | 9 | 1 | 0 |
| classification | 970/1000 | 97.0 | 8 | 2 | 0 |
| timeseries | 880/1000 | 88.0 | 8 | 2 | 0 |
| vision | 1000/1000 | 100.0 | 10 | 0 | 0 |

## 항목별 KPI

| 항목 | 점수 | 비율 |
|---|---:|---:|
| problem_understanding | 400/400 | 100.0% |
| data_leakage_guard | 600/600 | 100.0% |
| correct_validation | 555/600 | 92.5% |
| metric_understanding | 380/400 | 95.0% |
| baseline | 400/400 | 100.0% |
| modeling | 300/400 | 75.0% |
| failure_analysis | 400/400 | 100.0% |
| verifier_challenger | 400/400 | 100.0% |
| submission | 200/200 | 100.0% |
| human_output | 200/200 | 100.0% |

## REVIEW 5건

- **bike-sharing-demand**: 85점 / 부족=correct_validation; inferred split=group-aware; metric runtime=EXACT_PROXY
- **spaceship-titanic**: 85점 / 부족=correct_validation; inferred split=group-aware; metric runtime=EXACT_PROXY
- **ieee-fraud-detection**: 85점 / 부족=correct_validation; inferred split=group-aware; metric runtime=EXACT_PROXY
- **m5-forecasting-accuracy**: 80점 / 부족=metric_understanding, modeling; inferred split=rolling-origin; metric runtime=SPEC_KNOWN_RUNTIME_APPROX
- **covid19-global-forecasting-week-5**: 80점 / 부족=metric_understanding, modeling; inferred split=chronological; metric runtime=SPEC_KNOWN_RUNTIME_APPROX

### 해석

- Bike Sharing과 IEEE Fraud에서는 시간 순서와 Group 후보가 동시에 있을 때 Group split을 먼저 고르는 경향이 드러났음. 다음 버전은 `time + group` 복합 검증을 설계해야 함.
- Spaceship Titanic은 Group 구조를 보수적으로 우선해 benchmark 기준 stratified와 불일치함. 무조건 오류라기보다 split 선택 근거/목적을 더 명확히 해야 함.
- M5의 WRMSSE와 COVID Week 5의 weighted pinball은 대회 전용 가중/계층 정보를 완전히 재현하지 못해 REVIEW로 유지함.
- Time-Series 10문제에서는 proxy의 persistence baseline이 후보 모델보다 모두 강했음. 복잡한 모델을 억지로 승격하지 않는 건 맞지만, 시계열 모델링 능력 자체는 더 강화할 필요가 있음.
- Vision 10/10 100점은 proxy 이미지가 쉬웠기 때문이며 실제 Vision 성능 100%를 뜻하지 않음.

## Benchmark 명세 주의

- 40개는 실제 Kaggle competition identity를 사용했지만, 이번 세션에서 모든 40개 metric/submission schema를 공식 페이지별로 독립 검증한 것은 아님.
- 따라서 현재 catalog는 **내부 MASTER_EVAL V1**으로 취급하고, 외부 공인 Benchmark로 승격하기 전 competition별 source audit가 필요함.

## 데이터 접근 제한

- 현재 실행 환경에 Kaggle API credential(`~/.kaggle/kaggle.json`)이 없음.
- 따라서 40개 모두 `PROXY_DATA`로 실행함.
- 실제 Kaggle/MLE-bench 데이터가 준비되면 같은 Frozen Core로 재실행해야 진짜 Real-data KPI를 얻을 수 있음.

## 다음 개선 우선순위

1. Time-aware와 Group-aware를 단일 선택이 아니라 복합 Split 정책으로 설계
2. WRMSSE / weighted pinball 등 Kaggle 전용 Metric Adapter 구현
3. Time-Series rolling-origin + multi-seed + seasonal-naive + stronger model 비교
4. 실제 Kaggle/MLE-bench 데이터 40문제 실행
5. Vision benchmark 난이도 강화: source/group leakage, background bias, distribution shift
6. Human-friendly Notebook Renderer를 active Skill contract로 승격

## 평가 원칙

- Core Freeze 이후 Benchmark 실패를 보고 V6.1 코드는 수정하지 않았음.
- `MASTER_EVAL_RESULTS_V6_STRICT.json`에 40문제 원본 결과를 보존함.
- 실제 Kaggle 점수와 Proxy KPI를 섞지 않음.
