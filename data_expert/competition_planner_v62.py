from __future__ import annotations

from typing import Any
import pandas as pd

from competition_planner_v6 import CompetitionPlannerV6


class CompetitionPlannerV62(CompetitionPlannerV6):
    """V6.2 supports compound temporal+group validation instead of precedence-only routing."""

    def inspect(self, spec, df: pd.DataFrame | None = None) -> dict[str, Any]:
        plan = super().inspect(spec, df)
        groups = list(plan.get("group_candidates") or [])
        times = list(plan.get("time_candidates") or [])
        risks = {str(r).lower() for r in plan.get("risk_flags", [])}
        requested = str(plan.get("requested_validation") or "").lower()

        time_required = (
            requested in {"time-aware", "chronological", "rolling-origin"}
            or any("time" in r for r in risks)
            or plan.get("category") == "timeseries"
        )
        group_required = (
            requested == "group-aware"
            or any("group" in r or "entity" in r or "driver" in r or "building" in r for r in risks)
        )
        if df is not None and groups and times and time_required and group_required:
            plan["inferred_validation"] = "temporal-group"
            plan["validation_components"] = ["chronological", "group-isolation"]
            plan["validation_rationale"] = "Both temporal ordering and repeated-entity leakage are material."
        elif df is not None and groups and group_required and requested == "stratified":
            plan["inferred_validation"] = "stratified"
            plan["validation_components"] = ["stratified", "group-sensitivity-check"]
            plan["validation_rationale"] = "Primary metric estimation stays stratified; repeated groups require a secondary leakage sensitivity check."
        elif plan.get("inferred_validation") == "rolling-origin":
            plan["validation_components"] = ["rolling-origin"]
        elif plan.get("inferred_validation") == "chronological":
            plan["validation_components"] = ["chronological"]
        elif plan.get("inferred_validation") == "group-aware":
            plan["validation_components"] = ["group-isolation"]
        else:
            plan["validation_components"] = [plan.get("inferred_validation")]

        if plan.get("metric") == "wrmsse":
            plan["metric_adapter"] = "wrmsse"
            plan["metric_runtime"] = "EXACT_IF_SCALE_AND_WEIGHT_ARTIFACTS"
        elif plan.get("metric") == "pinball":
            plan["metric_adapter"] = "weighted_pinball"
            plan["metric_runtime"] = "EXACT_IF_QUANTILES_AND_WEIGHTS_AVAILABLE"
        return plan
