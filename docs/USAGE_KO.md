# AI DATA EXPERT V6.1 Candidate 사용 설명서

## 1. 무엇을 하는 도구인가

AI DATA EXPERT는 Codex에서 데이터 분석/ML/DL/Vision/Time-Series/MLOps/Big Data 작업을 요청할 때 사용하는 Repository Skill + Expert Harness입니다.

모델부터 돌리지 않고 먼저 다음을 확인합니다.

1. 관측 단위와 Target이 무엇인가
2. Target missing은 label 부재인가 새로운 상태인가
3. 예측 시점에 Feature를 실제 사용할 수 있는가
4. ID/row order/group/time leakage가 있는가
5. 어떤 split이 실제 일반화 조건과 맞는가
6. Baseline보다 복잡한 모델이 실제로 나은가
7. final Test가 선택 과정에서 오염되지 않았는가
8. minority/segment/OOD failure가 숨겨져 있지 않은가
9. 반대 가설을 검토했는가
10. PASS/REVIEW/FAIL 중 어떤 결론이 정당한가

Competition task에서는 metric/direction/submission contract도 먼저 고정합니다.

## 2. Windows 설치

```cmd
cd /d <AI_DATA_EXPERT 폴더>
setup_windows.bat
```

직접 설치:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

선택: Embedding + FAISS RAG

```cmd
setup_rag_embeddings_windows.bat
```

## 3. Codex에서 사용

프로젝트 루트에서:

```cmd
codex
```

예:

```text
이 CSV를 분석해줘.
Target 의미, 결측, 누수, 적절한 split, baseline, 실패 segment,
운영 적용 위험까지 확인해줘.
```

Notebook 과제:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv야.
```

## 4. Harness 직접 실행

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "수율 예측 + 데이터 품질 + 누수 + 검증 전략 검토" ^
  --target yield_percentage ^
  --out outputs\expert_context.json
```

업무 맥락을 알고 있으면 명시합니다.

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv data.csv ^
  --task "defect prediction" ^
  --target defect ^
  --prediction-time "after camera inspection before eject" ^
  --business-cost "false negative is much more expensive" ^
  --out outputs\result.json
```

## 5. Time-Series

시간 컬럼과 horizon을 가능한 명시합니다.

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

반대로:

```text
historical time-series analysis only; do not forecast
```

처럼 명시하면 forecast-only verifier를 요구하지 않는 것이 V6.1 계약입니다.

`solve_timeseries_rnn_v5.py`는 현재 포함된 Steel Industry 연습 데이터 구조를 위한 전용 notebook solver입니다. 범용 시계열 solver가 아닙니다.

## 6. Domain RAG

`domain_knowledge/` 또는 `--domain-path`를 사용합니다.

추천 문서:

```text
data_dictionary.md
process_flow.md
sensor_spec.md
quality_standard.md
defect_definition.md
incident_history.md
operational_constraints.json
```

검색된 근거는 source/provenance를 유지해야 하며, structured fact가 있을 때만 prediction time / group / unavailable feature / cost를 TaskSpec에 반영합니다.

## 7. Notebook 도구

구조 확인:

```cmd
python .agents\skills\ai-data-expert\scripts\inspect_notebook.py question.ipynb
```

일반 tabular solver:

```cmd
python .agents\skills\ai-data-expert\scripts\solve_notebook.py ^
  --input question.ipynb ^
  --data data.csv ^
  --output answer.ipynb
```

검증:

```cmd
python .agents\skills\ai-data-expert\scripts\validate_notebook.py answer.ipynb --timeout 300
```

## 8. Competition-aware planning

V6 계층은 CompetitionSpec을 통해 다음을 고정합니다.

- target
- metric
- metric direction
- category
- validation contract
- risk flags
- submission mode
- exact metric runtime 가능 여부

복잡한 competition metric이 hierarchy/weight 같은 원본 artifact를 필요로 하는데 파일이 없으면 generic RMSE로 바꿔치기하지 않고 `SPEC_KNOWN_RUNTIME_APPROX` / `REVIEW`로 남깁니다.

## 9. 결과 상태

### PASS
현재 evidence/contract에서 치명적 위반이 발견되지 않음.

### REVIEW
분석은 가능하지만 중요한 uncertainty/assumption이 남음.

예:
- prediction time UNKNOWN
- rare positive support 부족
- inferred group/run
- competition metric artifact 부족
- model이 baseline을 안정적으로 이기지 못함

### FAIL
- 실행 실패
- leakage/validation contract 위반
- 필요한 expert/verification 실패

## 10. 현재 추천 용도

적합:
- EDA / Data Quality Review
- Tabular Regression / Classification
- 불균형 분류 검토
- Notebook 과제/실습 검증
- PyTorch Tabular DL / Vision 검토
- Time-Series 설계 및 sequence exercise 검토
- Competition metric/validation planning
- MLOps Drift review
- Big Data architecture review

아직 자동 Production 승인, 자동 재학습, 무검토 인과 결론, 실제 Kaggle leaderboard 성능 보장은 하지 않습니다.
