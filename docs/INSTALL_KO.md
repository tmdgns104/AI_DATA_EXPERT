# AI Data Expert V4 설치 가이드 (Windows)

목표는 **Repository를 받은 뒤 추가 파일 복사나 수동 설정 없이 바로 사용할 수 있는 상태**입니다.

## 사전 조건

- Windows 10/11
- Python 3.11+ 권장
- Git은 clone 방식에서만 필요
- Codex를 상위 Agent로 사용할 경우 Codex CLI가 PATH에 있어야 함

Codex가 없어도 `run_expert.py`, `solve_notebook.py`, `run_demo_without_codex.bat`는 직접 사용할 수 있습니다.

## 방법 A — Git clone

```cmd
git clone https://github.com/tmdgns104/AI_DATA_EXPERT.git
cd AI_DATA_EXPERT
setup_windows.bat
```

## 방법 B — GitHub ZIP 다운로드

1. GitHub Repository에서 **Code → Download ZIP**
2. 원하는 폴더에 압축 해제
3. 해당 폴더에서 CMD 실행
4. 실행:

```cmd
setup_windows.bat
```

## setup_windows.bat가 자동으로 하는 일

1. UTF-8 / Jupyter 로컬 디렉터리 설정
2. `.venv` 생성
3. pip 업그레이드
4. `requirements.txt` 전체 설치
5. deterministic demo CSV 생성
6. `verify_install.py` 실행
7. V4 Runtime import
8. 실제 작은 분류 문제 실행
9. `data-analyst → machine-learning` Route 확인
10. Runtime Verifier가 `FAIL`이 아닌지 확인

중간 단계가 하나라도 실패하면 `Setup FAILED`와 non-zero exit code로 종료합니다.

성공 시 마지막에 다음 메시지가 나옵니다.

```text
AI Data Expert V4 is ready.
```

Codex CLI도 설치되어 있으면:

```text
Setup verified. Start Codex in this folder with: codex
```

이 표시됩니다.

## Codex로 바로 사용

설치 성공 후 같은 폴더에서:

```cmd
codex
```

예시 요청:

```text
examples/DNN_regression_question.ipynb 문제를 풀어서
outputs/my_answer.ipynb 로 만들어줘.
데이터는 examples/4_manufacturing_yield.csv 를 사용해.
```

Repository의 `AGENTS.md`와 `.agents/skills/ai-data-expert/SKILL.md`가 데이터/ML/DL/Vision/시계열/Big Data/MLOps/Notebook 작업에 AI Data Expert Skill을 사용하도록 안내합니다.

## Codex 없이 바로 테스트

```cmd
run_demo_without_codex.bat
```

또는 Expert Harness 직접 실행:

```cmd
.venv\Scripts\python.exe .agents\skills\ai-data-expert\scripts\run_expert.py ^
  --csv examples\4_manufacturing_yield.csv ^
  --task "yield_percentage를 예측하고 누수와 평가 방법까지 검토해줘" ^
  --target yield_percentage ^
  --prediction-time "before process completion" ^
  --out outputs\expert_context.json
```

## Embedding + FAISS RAG (선택)

기본 설치만으로 Hybrid RAG의 BM25 + offline vector fallback이 동작합니다.

실제 sentence-transformers + FAISS 경로를 사용하려면:

```cmd
setup_rag_embeddings_windows.bat
```

단, embedding model을 실제 사용할 로컬/다운로드 환경이 필요합니다. V4의 기존 Sandbox 검증은 offline fallback 기준이며 Embedding+FAISS 품질 평가는 다음 연구 과제로 남아 있습니다.

## 문제가 생기면

먼저:

```cmd
.venv\Scripts\python.exe verify_install.py
```

을 실행합니다.

`stage`가 다음 중 어디인지 확인하면 됩니다.

- `dependencies`: Python 패키지 설치 문제
- `skill`: Codex Skill 파일 누락
- `runtime_import`: V4 Runtime 파일/의존성 문제
- `runtime_smoke`: 실제 Expert 실행 문제

`verify_install.py`가 `PASS`이면 기본 Runtime은 사용할 수 있는 상태입니다.
