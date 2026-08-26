# AI DATA EXPERT

**Codex에서 데이터 분석·ML/DL·Vision·Time-Series 작업을 더 안전하게 수행하기 위한 검증형 Repository Skill + Expert Harness**

- 최신 기준: **V6.5 Frozen Candidate**
- 상태: **Frozen Candidate / Pre-Production Expert Copilot**
- 분석 코어 회귀: **58/58 PASS**
- Notebook 출력 스타일 회귀: **2/2 PASS**
- 설치 smoke: **PASS**
- 새로운 Kaggle 문제 명세 40개: **40/40 실제 모델 실행**
- 그중 baseline 상회: **34/40**
- 실제 Kaggle 원본 데이터 coverage: **0/40** — leaderboard 점수 아님

> 목표는 점수를 무조건 높이는 AutoML이 아니라, **문제 정의·누수·검증·근거·결론을 의심하고 사람이 이해하기 쉬운 결과로 정리하는 Data Expert**를 만드는 것입니다.

## 현재 흐름

```text
데이터/문제 확인
  ↓
TaskSpec + Data/Leakage Guard
  ↓
Domain RAG
  ↓
Intent / Modality / Competition Planning
  ↓
Experts + Shared Evidence + Argument
  ↓
Experiment / Baseline / Validation
  ↓
Challenger + Verifier
  ↓
Final Test
  ↓
Human-friendly Notebook / Report
```

## V6.5 핵심 개선

- 처음 보는 `crew_ref`, `family_bundle`, `merchant_cohort` 같은 반복 Entity 후보 일반화
- timestamp가 증가하더라도 중간 간격이 깨지는 cadence break 탐지
- `target` 직접 복사 및 affine target proxy leakage 탐지
- Persistence/SeasonalNaive를 포함한 baseline-inclusive Champion 선택
- 내부 분석은 엄격하게 유지하면서 최종 Notebook은 자연스러운 한국어로 단순화
- `데이터 확인 → 이상 발견 → 근거 확인 → 판단 → 분석` 흐름을 출력 규칙으로 고정
- 32-step/threshold/lag처럼 과제가 정하지 않은 값은 가정으로 표현
- Validation은 선택, Test는 최종 보고에 사용
- MAE/RMSE/R²가 엇갈리면 그대로 해석
- RNN/LSTM의 parameter count가 다르면 구조 자체의 절대 우위로 과장하지 않음

Notebook 출력 기준: [OUTPUT_STYLE_CONTRACT_KO.md](.agents/skills/ai-data-expert/OUTPUT_STYLE_CONTRACT_KO.md)

## 새로운 Kaggle 40문제 평가

기존 MASTER_EVAL의 40문제를 재사용하지 않고 새로운 문제 명세를 40개 구성했습니다.

| 유형 | 실행 | Baseline 상회 |
|---|---:|---:|
| 회귀 | 10/10 | 6/10 |
| 분류 | 10/10 | 9/10 |
| 시계열 | 10/10 | 9/10 |
| Vision | 10/10 | 10/10 |
| **전체** | **40/40** | **34/40** |

중요: Kaggle 자격증명이 없어 **원본 competition train/test 데이터는 0/40**입니다. 각 문제 명세에 대해 modality가 맞는 실제 오프라인 데이터로 모델 학습/검증/Test를 직접 실행한 결과이며 **Kaggle leaderboard 점수가 아닙니다.**

상세: [KAGGLE_NEW40_REALDATA_V6_5_REPORT_KO.md](KAGGLE_NEW40_REALDATA_V6_5_REPORT_KO.md)

## 빠른 시작

```cmd
git clone https://github.com/tmdgns104/AI_DATA_EXPERT.git
cd AI_DATA_EXPERT
setup_windows.bat
python examples\generate_demo_data.py
codex
```

Codex 없이 smoke 확인:

```cmd
python verify_install.py
```

## 핵심 상태 파일

- `V6_5_CORE_FREEZE.json` — 분석 코어 Freeze
- `FINAL_STATUS_V6_5.json` — 현재 상태
- `TEST_STATUS_V6_5.json` — 테스트 증거
- `RELEASE_MANIFEST_V6_5.json` — 배포 Manifest
- `evaluation/kaggle_master_eval/new40_v65/NEW40_REALDATA_RESULTS_V65_V2.json` — 새 40문제 전체 결과
- `.agents/skills/ai-data-expert/SKILL.md` — Codex Skill

## 현재 한계

- 실제 Kaggle 원본 데이터/leaderboard 검증은 아직 없음
- 시계열은 rolling/multi-seed 안정성을 더 강화할 여지가 있음
- 의미 기반 group inference는 앞으로 처음 보는 명명 규칙에서 계속 검증 필요
- Vision 실제 카메라/배경/장비 shift 평가는 더 어려운 실데이터가 필요

과거 Freeze와 실패 기록은 덮어쓰지 않고 provenance로 보존합니다.
