# Expert Reasoning Contract V1

모든 Expert는 최종 답만 제출하지 않는다. 아래 검토기록을 Evidence와 함께 남긴다.

1. **UNDERSTAND** — 문제를 전문분야 언어로 다시 정의하고 목적/성공 조건을 명시
2. **INSPECT** — 실제 데이터/메타데이터/운영조건의 관찰 사실을 기록하고 추측과 구분
3. **QUESTION** — 누수, 사용 가능 시점, 품질, 운영 제약 등 결정 전 핵심 질문 생성
4. **HYPOTHESIZE** — 하나의 설명만 고르지 않고 경쟁 가설 유지
5. **TEST** — 데이터/통계/실험으로 가설을 검증하고 Test set을 모델 선택에 반복 사용하지 않음
6. **COMPARE** — Baseline 또는 대안을 동일 기준으로 비교
7. **DECIDE** — 선택과 이유를 Evidence에 연결
8. **CHALLENGE** — 자신의 결론이 틀릴 수 있는 반례/shortcut 탐색
9. **RISK** — 해결되지 않은 위험과 추가 필요 데이터 기록
10. **CONFIDENCE** — HIGH / MEDIUM / LOW 및 이유 기록

## 금지

- 모델 이름부터 정하고 데이터에 끼워 맞추기
- 성능 숫자만으로 배포 판단
- Feature importance를 인과관계로 단정
- 이미지 단위 Random Split을 무조건 사용
- 시계열 Random Split
- Drift 하나만 보고 자동 재학습
