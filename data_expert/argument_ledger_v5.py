from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ArgumentNode:
    id: str
    question: str
    hypotheses: list[str]
    required_evidence: list[str]
    observations: list[Any] = field(default_factory=list)
    counterarguments: list[str] = field(default_factory=list)
    decision: str | None = None
    status: str = "OPEN"
    confidence: str = "UNKNOWN"
    provenance: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)


class ArgumentLedger:
    VALID_STATUS = {"OPEN", "SUPPORTED", "REJECTED", "INCONCLUSIVE", "BLOCKED"}

    def __init__(self):
        self.nodes: list[ArgumentNode] = []

    def add(self, **kwargs: Any) -> ArgumentNode:
        node = ArgumentNode(**kwargs)
        if node.status not in self.VALID_STATUS:
            raise ValueError(f"invalid argument status: {node.status}")
        self.nodes.append(node)
        return node

    def snapshot(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "open_count": sum(n.status in {"OPEN", "INCONCLUSIVE"} for n in self.nodes),
            "supported_count": sum(n.status == "SUPPORTED" for n in self.nodes),
            "rejected_count": sum(n.status == "REJECTED" for n in self.nodes),
            "blocked_count": sum(n.status == "BLOCKED" for n in self.nodes),
        }
