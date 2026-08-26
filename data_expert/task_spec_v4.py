from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from task_spec import TaskSpecBuilder as V3TaskSpecBuilder
from data_guard_v4 import analyze_dataframe


class TaskSpecBuilderV4:
    def __init__(self):
        self.base = V3TaskSpecBuilder()

    def build(self, problem: dict[str, Any]) -> dict[str, Any]:
        spec = self.base.build(problem)
        path = problem.get("data_path"); target = problem.get("profile", {}).get("target")
        guard = None
        if path and Path(path).exists() and Path(path).suffix.lower() == ".csv":
            try:
                df = pd.read_csv(path)
                guard = analyze_dataframe(df, target)
                ts = spec.get("timestamp")
                if ts in df.columns and pd.api.types.is_numeric_dtype(df[ts]) and not any(k in str(ts).lower() for k in ["timestamp","datetime","date"]):
                    spec["timestamp"] = None
            except Exception:
                guard = None
        if guard:
            spec["data_guard"] = guard
            spec["available_features"] = guard.get("safe_feature_columns", spec.get("available_features"))
            strategy = guard.get("group_strategy")
            if strategy:
                if strategy.get("type") == "column":
                    spec["group_id"] = strategy.get("column")
                    spec["split_strategy"] = f"group-aware split by {strategy.get('column')}"
                elif strategy.get("type") == "derived_reset_run":
                    spec["group_id"] = f"DERIVED_RUN({strategy.get('column')})"
                    spec["split_strategy"] = f"group-aware split by derived production run from {strategy.get('column')} resets"
            spec["label_state"] = {
                "labeled_count": guard.get("labeled_count"),
                "unlabeled_count": guard.get("unlabeled_count"),
                "target_missing_rate": guard.get("target_missing_rate"),
            }
            if guard.get("target_missing_count", 0) > 0:
                spec.setdefault("unknowns", []).append("unlabeled target rows require separate prediction path")
            if guard.get("drop_feature_columns"):
                spec["excluded_identifier_features"] = guard["drop_feature_columns"]
        return spec


def apply_domain_facts(spec: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = dict(spec); applied = []
    unavailable = set(spec.get("excluded_domain_features", []))
    for fact in facts:
        fact_target = fact.get("target")
        if fact_target and spec.get("target") and str(fact_target) != str(spec.get("target")):
            continue
        kind = str(fact.get("type", "")).lower()
        field = fact.get("field") or fact.get("feature")
        value = fact.get("value")
        if kind in {"prediction_time", "prediction-time"} and spec.get("prediction_time") == "UNKNOWN" and value:
            spec["prediction_time"] = value; applied.append(fact)
        elif kind in {"business_cost", "cost_matrix"} and spec.get("business_cost") == "UNKNOWN" and value:
            spec["business_cost"] = value; applied.append(fact)
        elif kind in {"feature_unavailable", "prediction_time_unavailable"} and field:
            unavailable.add(str(field)); applied.append(fact)
        elif kind in {"group_id", "entity_group"} and field and not spec.get("group_id"):
            spec["group_id"] = str(field); spec["split_strategy"] = f"group-aware split by {field} (domain evidence)"; applied.append(fact)
        elif kind in {"observation_unit"} and value and "domain confirmation required" in str(spec.get("observation_unit", "")):
            spec["observation_unit"] = value; applied.append(fact)
    if unavailable:
        spec["excluded_domain_features"] = sorted(unavailable)
        if isinstance(spec.get("available_features"), list):
            spec["available_features"] = [c for c in spec["available_features"] if c not in unavailable]
    unknowns = [u for u in spec.get("unknowns", []) if not (u == "prediction_time" and spec.get("prediction_time") != "UNKNOWN") and not (u == "business_cost" and spec.get("business_cost") != "UNKNOWN")]
    spec["unknowns"] = unknowns
    spec["domain_facts_applied"] = applied
    return spec, applied
