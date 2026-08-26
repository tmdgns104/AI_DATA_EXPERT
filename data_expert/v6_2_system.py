from __future__ import annotations

from pathlib import Path
from typing import Any

from v6_1_system import V61System
from domain_rag_v62 import HybridDomainRAGV62
from competition_planner_v62 import CompetitionPlannerV62
from time_series_dl_v62 import TimeSeriesDLExpertV62
from adaptive_experiment_v62 import AdaptiveExperimentPolicyV62


class V62System(V61System):
    def __init__(self, root: str | Path | None = None):
        super().__init__(root=root)
        self.rag = HybridDomainRAGV62(self.root / "domain_knowledge")
        self.competition_planner = CompetitionPlannerV62()
        self.engines["time-series"] = TimeSeriesDLExpertV62()
        self.adaptive_policy = AdaptiveExperimentPolicyV62()

    def run(self, problem: dict[str, Any]):
        result = super().run(problem)
        recs = self.adaptive_policy.recommend(result.get("task_spec", {}), result.get("argument_ledger", {}), result)
        result["adaptive_next_experiments"] = recs
        if recs:
            result.setdefault("shared_evidence", {}).setdefault("adaptive_policy", {})
            result["shared_evidence"]["adaptive_policy"] = {"recommendations": recs, "count": len(recs)}
        return result


EnhancedSystem = V62System
