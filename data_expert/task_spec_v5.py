from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from task_spec_v4 import TaskSpecBuilderV4


class TaskSpecBuilderV5:
    def __init__(self):
        self.base = TaskSpecBuilderV4()

    def build(self, problem: dict[str, Any]) -> dict[str, Any]:
        spec = self.base.build(problem)
        p = problem.get("profile", {})
        text = str(problem.get("task", "")).lower()
        modality = str(p.get("modality", spec.get("modality", "tabular"))).lower()
        path = problem.get("data_path")

        if modality == "time-series":
            spec["modality"] = "time-series"
            spec["problem_type"] = "forecasting"
            # Detect a timestamp column from the actual data when possible.
            if path and Path(path).exists():
                try:
                    df = pd.read_csv(path, nrows=200)
                    for candidate in [p.get("timestamp"), "date", "datetime", "timestamp", "time"]:
                        if candidate and candidate in df.columns:
                            spec["timestamp"] = candidate
                            break
                except Exception:
                    pass
            intents = set(spec.get("intent", {}).get("intents", []))
            if any(k in text for k in ["rnn", "lstm", "순환신경망", "simplernn", "예측", "forecast"]):
                intents.update({"TRAIN_MODEL", "COMPARE_MODELS", "FORECAST", "DL_TRAIN"})
                spec["intent"]["intents"] = sorted(intents)
                spec["intent"]["primary_intent"] = "FORECAST"
                spec["intent"]["confidence"] = max(float(spec["intent"].get("confidence", 0.0)), 0.98)
            spec["split_strategy"] = "chronological train/validation/test"
            spec["primary_metric"] = "RMSE (MAE and R2 secondary)"
            if spec.get("observation_unit", "").startswith("one row") or not spec.get("observation_unit"):
                spec["observation_unit"] = "one sequential time interval"
            if spec.get("prediction_time") == "UNKNOWN":
                # The exercise asks for prediction but does not state a horizon. Do not hide the assumption.
                spec["prediction_time"] = "next observation (one-step ahead; simulation assumption)"
                spec.setdefault("assumptions", []).append("forecast horizon inferred as one-step ahead because the exercise does not specify it")
                spec.setdefault("unknowns", []).append("explicit forecast horizon")
            # Remove stale V3 unknown labels once V5 has established them from data/task evidence.
            spec["unknowns"] = [u for u in spec.get("unknowns", []) if u not in {"observation_unit", "group/entity id (if repeated entities exist)"}]
        return spec
