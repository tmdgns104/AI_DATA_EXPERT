# 실제 Codex 사용 검증 기록

기준일: **2026-08-26**

Benchmark 외에 실제 Notebook 작업을 Codex에서 수행하면서 얻은 Evidence를 기록합니다.

## Case A — 제조 상태 분류 / Partial Label

핵심 발견:

- Target missing은 제3 Class가 아니라 미라벨 예측 대상
- `_id`는 고유 식별자이면서 행 순서 Proxy
- Shot 번호 reset으로 여러 생산 run 후보 존재
- Random split과 Run-isolated split 사이 큰 성능 차이

실제 비교:

| 평가 | 결과 |
|---|---:|
| Random holdout macro-F1 | 약 0.907 |
| Run-isolated Test macro-F1 | 약 0.771 |
| Run-isolated status 1 recall | 약 0.381 |

판단:

- 높은 Random split 점수는 일반화 성능을 과대평가할 가능성이 있음
- Run 정의는 실제 MES/Batch ID가 아닌 추론값이므로 `REVIEW`
- 이 결과가 V4 Data Guard / Group split 개선의 직접 근거가 됨

## Case B — DNN Classification / Extreme Imbalance

데이터:

- 총 3,000행
- defect 9건 (0.3%)

접근:

- Stratified Train/Validation/Test
- Dummy / Balanced Logistic baseline
- PyTorch weighted-loss DNN
- small-batch overfit sanity check
- Validation checkpoint
- Validation threshold
- Final Test 1회
- defect recall / PR-AUC / Macro-F1 중심 평가

결과:

| 지표 | 결과 |
|---|---:|
| DNN Test Macro-F1 | 0.5246 |
| Balanced Accuracy | 0.7291 |
| Defect Recall | 0.5000 |

판정:

- Notebook 실행 및 의미 검증: `PASS`
- 운영 적용: `REVIEW`

이유:

- 전체 defect가 9건뿐임
- Validation/Test에서 defect 수가 매우 작음
- 하나의 오분류가 Recall을 크게 변화시킴
- 실제 FP/FN 비용과 batch/time metadata가 없음

## Case C — Vision AI 3-class CNN

실제 로컬 이미지 Dataset을 사용한 Pixel training.

검증:

- Train/Validation/Test split
- 완전 중복 이미지 검사
- PyTorch CNN 실제 학습
- best-validation checkpoint
- 모델 저장
- `weights_only=True` 방식의 안전 재로딩 검증
- clean notebook re-execution

결과:

| 지표 | 결과 |
|---|---:|
| Test Accuracy | 0.9444 |
| Balanced Accuracy | 0.9444 |
| Macro-F1 | 0.9441 |
| Errors | 4 / 72 |
| Complete duplicate across splits | 0 |

판정:

- 실행/의미 검증: `PASS`
- synthetic/local practice dataset이므로 실제 현장 성능 주장 금지

## 실제 사용에서 새로 발견한 시스템 마찰

1. Windows cp949/Jupyter 사용자 폴더 권한/출력 인코딩
2. Vision Expert와 Deep Learning Expert의 실행 상태 불일치 가능성
3. Tabular 중심 Semantic Validator가 Vision folder split을 완전히 이해하지 못함
4. Vision dependency (`torchvision`) 및 검증 dependency (`PyYAML`) 완비 필요
5. Notebook 생성기의 한글 문자열 품질 점검 필요

## 현재 실사용 판단

현재 프로젝트는 `완전 자율 분석가`보다 다음에 더 적합합니다.

> 분석가가 놓치기 쉬운 누수, Split, 희귀 Class, Test 오염, Domain 불확실성을 구조적으로 검토하는 Expert Copilot.

다음 개선은 이 실사용 기록을 V5 Acceptance Test로 고정한 뒤 진행합니다.
