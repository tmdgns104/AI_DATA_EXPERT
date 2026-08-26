# AI Data Expert V6.1 Candidate 설치 가이드 (Windows)

현재 `main` 기준은 **V6.1 Candidate**입니다. Production Release로 승격된 버전은 아닙니다.

## 사전 조건

- Windows 10/11
- Python 3.11+ 권장
- Git은 clone 방식에서만 필요
- Codex를 상위 Agent로 사용할 경우 Codex CLI가 PATH에 있어야 함

Codex가 없어도 `run_expert.py`, `solve_notebook.py`, `verify_install.py`는 직접 사용할 수 있습니다.

## 방법 A — Git clone

```cmd
git clone https://github.com/tmdgns104/AI_DATA_EXPERT.git
cd AI_DATA_EXPERT
setup_windows.bat
```

## 방법 B — GitHub ZIP

1. Repository에서 **Code → Download ZIP**
2. 압축 해제
3. 해당 폴더에서 CMD 실행
4. 실행:

```cmd
setup_windows.bat
```

## setup_windows.bat가 하는 일

현재 V6.1 Candidate의 `setup_windows.bat`는 다음만 수행합니다.

1. UTF-8 환경 설정
2. `.venv` 생성
3. 가상환경 활성화
4. pip 업그레이드
5. `requirements.txt` 설치
6. 설치 실패 시 non-zero exit code 반환

성공 시:

```text
Setup complete.
Start Codex in this folder with: codex
```

이 출력됩니다.

> 과거 V4 설치 문서에는 demo CSV 생성과 runtime 검증까지 `setup_windows.bat`가 자동으로 수행한다고 적혀 있었지만, 현재 V6.1 Candidate의 실제 script는 dependency setup까지만 수행합니다. 문서는 실제 동작에 맞춰 수정했습니다.

## 설치 검증

설치 후 별도로 실행합니다.

```cmd
.venv\Scripts\python.exe verify_install.py
```

`PASS`이면 기본 Skill/Runtime import와 smoke validation을 사용할 수 있는 상태입니다.

## Demo 데이터 생성

Repository에는 생성 script가 있으므로 예제 CSV가 필요하면 먼저 실행합니다.

```cmd
.venv\Scripts\python.exe examples\generate_demo_data.py
```

생성 후 다음과 같은 예제 경로를 사용할 수 있습니다.

```text
examples\4_manufacturing_yield.csv
examples\classification_example.csv
```

## Codex로 사용

```cmd
codex
```

예시:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/my_answer.ipynb로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv를 사용해.
```

Repository의 `AGENTS.md`와 `.agents/skills/ai-data-expert/SKILL.md`가 현재 V6.1 계약을 설명합니다.

## Harness 직접 실행

Demo CSV를 생성한 뒤:

```cmd
.venv\Scripts\python.exe .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "yield_percentage를 예측하고 누수와 평가 방법까지 검토해줘" ^
  --target yield_percentage ^
  --prediction-time "before process completion" ^
  --out outputs\expert_context.json
```

Time-Series 예시:

```cmd
.venv\Scripts\python.exe .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv series.csv ^
  --task "forecast the next 24 hours" ^
  --target target ^
  --modality time-series ^
  --timestamp-col timestamp ^
  --horizon 24h ^
  --out outputs\forecast_context.json
```

## Embedding + FAISS RAG (선택)

기본 환경에서는 Hybrid RAG가 사용 가능한 fallback 경로를 사용할 수 있습니다.

실제 sentence-transformers + FAISS 경로를 준비하려면:

```cmd
setup_rag_embeddings_windows.bat
```

단, V6.1의 현재 상태 문서에서 Embedding+FAISS runtime/품질 검증은 환경 의존 항목으로 남아 있습니다. 설치됐다는 이유만으로 품질 검증까지 완료된 것으로 보지 않습니다.

## 문제가 생기면

먼저:

```cmd
.venv\Scripts\python.exe verify_install.py
```

을 실행하고 `stage`를 확인합니다.

- `dependencies`: Python 패키지 문제
- `skill`: Repository Skill 파일 문제
- `runtime_import`: Runtime import/의존성 문제
- `runtime_smoke`: 실제 Expert smoke 실행 문제

## 현재 버전 상태

- main baseline: `V6.1_CANDIDATE`
- freeze-time recorded regression: `44/44 PASS`
- MASTER_EVAL V1 proxy: `95.875`
- 실제 Kaggle 원본 데이터 coverage: `0/40`

위 수치는 Repository의 Freeze/KPI 증거 기준입니다. 설치 직후 이 문서만 보고 새 환경의 전체 44개 테스트가 자동으로 재실행됐다고 해석하면 안 됩니다.
