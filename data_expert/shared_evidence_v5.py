from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EvidenceRecord:
    key: str
    value: Any
    source: str
    confidence: str = "MEDIUM"
    status: str = "OBSERVED"


class SharedEvidenceStore:
    """Small in-process evidence ledger shared by all V5 experts/verifiers."""

    def __init__(self):
        self._records: list[EvidenceRecord] = []

    def publish(self, key: str, value: Any, source: str, *, confidence: str = "MEDIUM", status: str = "OBSERVED") -> None:
        self._records.append(EvidenceRecord(key, value, source, confidence, status))

    def publish_many(self, mapping: dict[str, Any], source: str, *, confidence: str = "MEDIUM") -> None:
        for key, value in mapping.items():
            self.publish(key, value, source, confidence=confidence)

    def latest(self, key: str, default: Any = None) -> Any:
        for record in reversed(self._records):
            if record.key == key:
                return record.value
        return default

    def snapshot(self) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for record in self._records:
            latest[record.key] = record.value
        return {
            "latest": latest,
            "records": [asdict(r) for r in self._records],
            "record_count": len(self._records),
        }
