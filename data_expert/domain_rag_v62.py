from __future__ import annotations

from pathlib import Path
from typing import Any

from domain_rag_v5 import HybridDomainRAGV5


class HybridDomainRAGV62(HybridDomainRAGV5):
    """V6.2 tightens negative relevance and fact scoping.

    A retrieved chunk is not automatically eligible evidence. Bundled synthetic demo
    knowledge must match the active target/domain, and structured facts must be scoped
    to the active target before they are allowed to mutate TaskSpec.
    """

    @staticmethod
    def _norm(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", " ").replace("_", " ")

    def retrieve(self, problem: dict[str, Any], top_k: int = 4) -> dict[str, Any]:
        raw = super().retrieve(problem, top_k=max(top_k * 2, 8))
        profile = problem.get("profile", {})
        target = self._norm(profile.get("target"))
        domain = self._norm(profile.get("domain"))
        process = self._norm(profile.get("process"))
        modality = self._norm(profile.get("modality"))
        query_terms = {x for x in [target, domain, process, modality] if x}

        accepted = []
        rejected = list(raw.get("rejected_matches", []))
        accepted_sources: set[str] = set()
        for match in raw.get("matches", []):
            source = str(match.get("source", ""))
            source_l = source.lower()
            text = self._norm(f"{match.get('heading','')} {match.get('text','')}")
            score = float(match.get("score", 0.0))
            bundled_demo = "domain_knowledge" in source_l and any(
                token in source_l or token in text
                for token in ["manufacturing_example", "synthetic demo", "example knowledge", "bundled demo"]
            )
            target_hit = bool(target and target in text)
            context_hit = any(term in text for term in query_terms if len(term) >= 3)

            if bundled_demo and target and not target_hit:
                rejected.append({"source": source, "reason": "bundled_demo_target_mismatch"})
                continue
            if target and not target_hit and not context_hit and score < 0.82:
                rejected.append({"source": source, "reason": "insufficient_task_relevance"})
                continue
            accepted.append(match)
            accepted_sources.add(source)
            if len(accepted) >= top_k:
                break

        facts = []
        rejected_facts = []
        for fact in raw.get("facts", []):
            ftarget = self._norm(fact.get("target"))
            fsource = str(fact.get("source", ""))
            if ftarget and target and ftarget != target:
                rejected_facts.append({"source": fsource, "type": fact.get("type"), "reason": "fact_target_mismatch"})
                continue
            if fsource and accepted_sources and fsource not in accepted_sources:
                rejected_facts.append({"source": fsource, "type": fact.get("type"), "reason": "fact_source_not_accepted"})
                continue
            if fsource and not accepted_sources:
                rejected_facts.append({"source": fsource, "type": fact.get("type"), "reason": "no_accepted_evidence_source"})
                continue
            facts.append(fact)

        raw["matches"] = accepted
        raw["facts"] = facts
        raw["rejected_matches"] = rejected
        raw["rejected_facts"] = rejected_facts
        raw["status"] = "FOUND" if accepted or facts else "NO_MATCH"
        raw["evidence_gate"] = {
            "version": "V6.2",
            "accepted_matches": len(accepted),
            "rejected_matches": len(rejected),
            "accepted_facts": len(facts),
            "rejected_facts": len(rejected_facts),
        }
        return raw
