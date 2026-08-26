from __future__ import annotations

from typing import Any


class AdaptiveExperimentPolicyV62:
    """Turn unresolved arguments into bounded next experiments."""

    def recommend(self, task_spec: dict[str, Any], ledger: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        nodes = ledger.get("nodes", []) if isinstance(ledger, dict) else []
        by_id = {n.get("id"): n for n in nodes}
        model = by_id.get("H-MODEL-001", {})
        horizon = by_id.get("H-HORIZON-001", {})
        if model.get("status") in {"INCONCLUSIVE", "OPEN"}:
            recs.append({
                "id": "EXP-TS-ROLLING-001",
                "reason": "model ranking is not stable enough on one holdout",
                "experiment": "rolling-origin backtest across at least 3 origins",
                "priority": "HIGH",
            })
            recs.append({
                "id": "EXP-TS-SEED-001",
                "reason": "neural ranking may depend on initialization",
                "experiment": "repeat recurrent candidates across multiple random seeds",
                "priority": "MEDIUM",
            })
        if horizon.get("status") == "INCONCLUSIVE" and "explicit forecast horizon" in task_spec.get("unknowns", []):
            recs.append({
                "id": "EXP-HORIZON-BLOCKED-001",
                "reason": "business/assignment horizon is unknown",
                "experiment": "do not promote one-step result as universal; request/resolve horizon before deployment",
                "priority": "HIGH",
                "status": "BLOCKED_BY_REQUIREMENT",
            })
        return recs[:4]
