from __future__ import annotations

from typing import Any
import numpy as np

from time_series_dl_v5 import TimeSeriesDLExpertV5, _metrics


class TimeSeriesDLExpertV62(TimeSeriesDLExpertV5):
    """V6.2: baselines participate in validation-time champion selection.

    Also adds a real seasonal-naive baseline when the dominant interval supports a
    daily lag, and reports model-vs-baseline margins instead of assuming a neural
    model must win.
    """

    agent = "time-series"

    def run(self, problem: dict[str, Any]) -> dict[str, Any]:
        out = super().run(problem)
        d = out["DECIDE"]
        path = problem.get("data_path")
        profile = problem.get("profile", {})
        target = profile.get("target", "Usage_kWh")

        import pandas as pd
        df = pd.read_csv(path)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        n = len(y); train_end = int(n * 0.70); val_end = int(n * 0.85)
        val_idx = np.arange(max(train_end, self.sequence_length), val_end)
        test_idx = np.arange(max(val_end, self.sequence_length), n)

        validation_metrics = dict(d.get("validation_metrics", {}))
        validation_metrics["Persistence"] = _metrics(y[val_idx], y[val_idx - 1])

        interval = d.get("timestamp_repair", {}).get("dominant_interval_seconds")
        seasonal_lag = None
        if interval and interval > 0:
            daily = int(round(86400.0 / float(interval)))
            if daily >= 2 and len(y) > daily + self.sequence_length:
                seasonal_lag = daily
                valid_val = val_idx[val_idx >= daily]
                valid_test = test_idx[test_idx >= daily]
                if len(valid_val):
                    validation_metrics["SeasonalNaive"] = _metrics(y[valid_val], y[valid_val - daily])
                    d.setdefault("test_metrics", {})["SeasonalNaive"] = _metrics(y[valid_test], y[valid_test - daily])

        neural_candidates = ["SimpleRNN", "LSTM"]
        all_candidates = list(validation_metrics)
        champion = min(all_candidates, key=lambda k: validation_metrics[k]["RMSE"])
        neural_winner = min(neural_candidates, key=lambda k: validation_metrics[k]["RMSE"])
        best_baseline = min([k for k in all_candidates if k not in neural_candidates], key=lambda k: validation_metrics[k]["RMSE"])
        baseline_rmse = validation_metrics[best_baseline]["RMSE"]
        neural_rmse = validation_metrics[neural_winner]["RMSE"]

        d["validation_metrics"] = validation_metrics
        d["selected_by_validation_rmse"] = champion
        d["selected_neural_model"] = neural_winner
        d["best_baseline"] = best_baseline
        d["neural_vs_best_baseline_rmse_gain_pct"] = float((baseline_rmse - neural_rmse) / baseline_rmse * 100.0)
        d["seasonal_lag"] = seasonal_lag
        d["champion_is_baseline"] = champion not in neural_candidates

        markers = set(out.get("markers", []))
        markers.update({"baseline_in_champion_selection", "validation_baseline_metrics"})
        if seasonal_lag is not None:
            markers.add("seasonal_naive_actual")
        out["markers"] = sorted(markers)
        out["QUESTION"] = "Which candidate, including simple baselines, is the validation-time champion?"
        out["CHALLENGE"] = list(out.get("CHALLENGE", [])) + [
            "A recurrent model is not promoted if a simple baseline wins validation RMSE.",
            "Seasonal naive is computed when the timestamp interval supports a daily lag; the marker is not emitted without an actual calculation.",
        ]
        return out
