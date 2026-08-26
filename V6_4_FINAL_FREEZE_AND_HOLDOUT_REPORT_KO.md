# V6.4 최종 Freeze + 독립 Holdout 요약

V6.4는 전체 회귀 **54/54 PASS** 후 Core를 Freeze했고, Freeze 이후 새로운 200개 구조 Holdout을 수행했습니다.

결과는 **80/200 = 40.0%**였습니다.

이 결과가 낮다고 해서 V6.4 코어를 다시 수정하지 않았습니다. 실패는 다음 버전 개선 항목으로 넘겼습니다.

주요 실패:
- 처음 보는 반복 Entity 이름 일반화 부족
- 시간은 증가하지만 cadence가 크게 깨지는 경우 탐지 부족
- target 직접 복사/affine post-outcome leakage 탐지 부족

V6.4에서 새로 고친 RAG domain negation 계열은 새로운 표현에서도 통과했습니다.
