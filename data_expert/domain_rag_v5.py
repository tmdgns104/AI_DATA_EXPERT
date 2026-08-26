from __future__ import annotations

from typing import Any
from domain_rag_v4 import HybridDomainRAG


class HybridDomainRAGV5(HybridDomainRAG):
    """V5 adds a wrong-domain injection guard on top of V4 hybrid retrieval."""

    def retrieve(self, problem: dict[str, Any], top_k: int = 4) -> dict[str, Any]:
        raw = super().retrieve(problem, top_k=max(top_k * 2, 6))
        target = str(problem.get("profile", {}).get("target") or "").strip().lower()
        modality = str(problem.get("profile", {}).get("modality") or "").strip().lower()
        filtered=[]; rejected=[]
        for m in raw.get("matches", []):
            text=(str(m.get("text", "")) + " " + str(m.get("heading", ""))).lower()
            source=str(m.get("source", "")).lower()
            # Generic repository README is instructions, not domain evidence.
            if source.endswith("domain_knowledge/readme.md"):
                rejected.append({"source":m.get("source"),"reason":"generic_domain_instructions_not_evidence"}); continue
            # When a concrete target is known, synthetic/demo evidence for another target must not be injected.
            target_hit = bool(target and target in text)
            modality_hit = bool(modality and modality.replace("-"," ") in text)
            is_synthetic_demo = "synthetic demo" in text or "example knowledge" in text
            if target and not target_hit and is_synthetic_demo:
                rejected.append({"source":m.get("source"),"reason":"target_mismatch_demo_evidence"}); continue
            # Weak fallback matches without target/modality support are safer as NO_MATCH than as evidence.
            if target and not target_hit and not modality_hit and float(m.get("score",0)) < 0.70:
                rejected.append({"source":m.get("source"),"reason":"weak_cross_domain_match"}); continue
            filtered.append(m)
            if len(filtered)>=top_k: break
        raw["matches"]=filtered
        raw["rejected_matches"]=rejected
        raw["status"]="FOUND" if filtered or raw.get("facts") else "NO_MATCH"
        raw["wrong_domain_guard"]={"enabled":True,"rejected_count":len(rejected)}
        return raw
