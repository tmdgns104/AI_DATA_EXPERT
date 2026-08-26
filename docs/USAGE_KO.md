# AI DATA EXPERT V4 사용 설명서

## 1. 목적

AI DATA EXPERT는 Codex에서 데이터 분석/ML/DL/Vision/Time-Series/MLOps/Big Data 작업을 요청할 때 자동으로 사용할 수 있는 Repository Skill + Expert Harness입니다.

일반적인 AutoML처럼 모델만 돌리지 않고 다음을 먼저 확인합니다.

1. 무엇을 예측/분석하는가
2. 한 행의 관측 단위는 무엇인가
3. Target의 의미와 결측은 무엇인가
4. 예측 시점에 실제 사용할 수 있는 Feature인가
5. Group/Entity/Time 구조가 있는가
6. 어떤 Split이 현실적인가
7. Baseline보다 실제로 나은가
8. Test set을 선택에 사용하지 않았는가
9. 실패하는 Class/Segment는 어디인가
10. 운영 적용을 PASS로 해도 되는가

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

### 선택: Embedding + FAISS RAG

```cmd
setup_rag_embeddings_windows.bat
```

`requirements-rag-optional.txt`에는 optional semantic retrieval dependency가 정리되어 있습니다.

## 3. Codex에서 사용

프로젝트 루트에서:

```cmd
codex
```

자연어로 요청합니다.

```text
이 CSV를 분석해서 Target을 예측해줘.
데이터 누수, 적절한 분할, baseline, 평가 방법과 운영 위험까지 확인해줘.
```

Notebook 과제:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv야.
```

Skill 이름을 직접 말하고 싶다면 `$ai-data-expert`를 명시해도 됩니다.

## 4. Harness 직접 실행

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "수율을 예측하고 데이터 품질, 누수, 평가 방식을 검토" ^
  --target yield_percentage ^
  --out outputs\expert_context.json
```

중요한 업무 맥락을 알고 있다면 명시하는 것이 좋습니다.

```cmd
python .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv data.csv ^
  --task "defect prediction" ^
  --target defect ^
  --prediction-time "after camera inspection before eject" ^
  --business-cost "false negative is much more expensive" ^
  --out outputs\result.json
```

## 5. Domain RAG 사용

`domain_knowledge/`에 조직/공정 지식을 넣습니다.

권장 문서:

```text
domain_knowledge/
├─ data_dictionary.md
├─ process_flow.md
├─ sensor_spec.md
├─ quality_standard.md
├─ defect_definition.md
├─ incident_history.md
└─ operational_constraints.json
```

Structured fact 예:

```json
{
  "facts": [
    {
      "type": "prediction_time",
      "target": "defect",
      "value": "after camera exposure before eject"
    },
    {
      "type": "group_id",
      "target": "defect",
      "field": "lot_code"
    },
    {
      "type": "feature_unavailable",
      "target": "defect",
      "field": "post_inspection_code"
    }
  ]
}
```

검색된 지식은 단순 참고 문장으로 끝나지 않고 TaskSpec과 Feature/Group/Prediction-time 판단에 연결됩니다.

## 6. Notebook 도구

Notebook 구조 확인:

```cmd
python .agents\skills\ai-data-expert\scripts\inspect_notebook.py question.ipynb
```

자동 Solver:

```cmd
python .agents\skills\ai-data-expert\scripts\solve_notebook.py ^
  --input question.ipynb ^
  --output answer.ipynb
```

실행/검증:

```cmd
python .agents\skills\ai-data-expert\scripts\validate_notebook.py answer.ipynb --timeout 300
```

## 7. 결과 상태 읽기

### PASS
현재 Evidence에서 치명적인 분석 계약 위반이 없음.

### REVIEW
다음과 같은 경우 정상적인 결과입니다.

- Prediction time UNKNOWN
- Business cost UNKNOWN
- 양성/불량 표본이 너무 적음
- Group/Run이 추론값임
- Domain evidence 부족
- 운영 적용에 필요한 metadata 부족

### FAIL

- Expert 실행 오류
- Test leakage
- Target/Feature 계약 위반
- 필요한 Expert 미실행
- 실행 결과가 분석 계약과 충돌

## 8. 추천 사용 방식

현재 V4는 다음 용도에 적합합니다.

- EDA / Data Quality Review
- Tabular Regression / Classification
- 불균형 분류 검토
- Notebook 과제/실습 초안 및 검증
- PyTorch Tabular DL
- Vision Dataset / CNN 학습 검토
- Time-Series 설계 검토
- Survival 분석 검토
- MLOps Drift 검토
- Big Data Architecture 검토

Production 자동 승인, 자동 재학습, 인과 결론은 사람 검토 없이 사용하지 않습니다.
