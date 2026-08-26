# Domain Knowledge

회사/프로젝트별 도메인 근거 문서를 이 폴더에 둡니다. Runtime은 관련 문서를 검색해 Evidence로 사용하지만 검색된 내용을 보편적 진리로 취급하지 않습니다.

권장 문서:
- Data Dictionary / 관측 단위 정의
- 센서 사양과 유효 범위
- 공정 순서와 prediction-time feature availability
- 품질/불량 정의와 FP/FN 업무 비용
- 장애/Incident 기록
- 배포 latency/memory/운영 제약

`.md`, `.txt`, `.json`을 지원하며 JSON의 structured facts는 prediction time, group id, feature eligibility, business cost 같은 TaskSpec 정보에 반영할 수 있습니다.
