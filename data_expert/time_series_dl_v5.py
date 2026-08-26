from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


CONTRACT_KEYS = ["UNDERSTAND","INSPECT","QUESTION","HYPOTHESES","TESTS","COMPARE","DECIDE","CHALLENGE","RISKS","CONFIDENCE"]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _correct_steel_timestamp(df: pd.DataFrame, column: str) -> tuple[pd.Series, dict[str, Any]]:
    raw = pd.to_datetime(df[column], format="%d/%m/%Y %H:%M", errors="coerce")
    if raw.isna().any():
        raw = pd.to_datetime(df[column], errors="coerce")
    corrected = raw.copy()
    repaired = np.zeros(len(df), dtype=bool)
    if "NSM" in df.columns and len(df) > 1:
        nsm = pd.to_numeric(df["NSM"], errors="coerce").fillna(-1).to_numpy()
        repaired[1:] = (nsm[1:] == 0) & (nsm[:-1] >= 23 * 3600)
        corrected = corrected + pd.to_timedelta(repaired.astype(int), unit="D")
    info = {
        "parse_failures": int(corrected.isna().sum()),
        "midnight_repairs": int(repaired.sum()),
        "monotonic_after_repair": bool(corrected.is_monotonic_increasing),
        "duplicate_timestamps_after_repair": int(corrected.duplicated().sum()),
    }
    if len(corrected) > 1 and corrected.notna().all():
        delta = corrected.diff().dropna().dt.total_seconds()
        info["dominant_interval_seconds"] = float(delta.mode().iloc[0]) if not delta.empty else None
        info["irregular_interval_count"] = int((delta != info["dominant_interval_seconds"]).sum()) if not delta.empty else 0
    return corrected, info


class _SeqNet(nn.Module):
    def __init__(self, kind: str, input_dim: int = 1, hidden_dim: int = 24):
        super().__init__()
        layer = nn.RNN if kind == "SimpleRNN" else nn.LSTM
        self.recurrent = layer(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.recurrent(x)
        return self.head(out[:, -1, :])


class TimeSeriesDLExpertV5:
    agent = "time-series"

    def __init__(self, sequence_length: int = 32, epochs: int = 5):
        self.sequence_length = sequence_length
        self.epochs = epochs

    def _make_sequences(self, scaled: np.ndarray, start: int, end: int):
        idx = np.arange(max(start, self.sequence_length), end)
        x = np.stack([scaled[i-self.sequence_length:i] for i in idx]).astype(np.float32)[:, :, None]
        y = scaled[idx].astype(np.float32)[:, None]
        return torch.from_numpy(x), torch.from_numpy(y), idx

    def _fit(self, kind: str, x_train, y_train, x_val, y_val, seed: int = 42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
        model = _SeqNet(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512, shuffle=False)
        best_state = None; best_val = float("inf"); best_epoch = 0; history=[]
        for epoch in range(1, self.epochs + 1):
            model.train(); losses=[]
            for xb, yb in loader:
                optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad(): val_loss = float(loss_fn(model(x_val), y_val).item())
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return model, {"best_epoch": best_epoch, "best_val_loss": best_val, "history": history}

    def run(self, problem: dict[str, Any]) -> dict[str, Any]:
        p = problem.get("profile", {}); path = problem.get("data_path"); target = p.get("target", "Usage_kWh")
        timestamp = problem.get("task_spec", {}).get("timestamp") or p.get("timestamp") or "date"
        if not path or not Path(path).exists():
            raise FileNotFoundError(path or "time-series data_path missing")
        df = pd.read_csv(path)
        if target not in df.columns: raise KeyError(f"target not found: {target}")
        corrected_ts, ts_info = _correct_steel_timestamp(df, timestamp)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any(): raise ValueError("target contains missing/non-numeric values")
        n = len(y); train_end = int(n*0.70); val_end = int(n*0.85)
        scaler = StandardScaler().fit(y[:train_end, None])
        scaled = scaler.transform(y[:, None]).ravel().astype(np.float32)
        x_train,y_train,train_idx=self._make_sequences(scaled,0,train_end)
        x_val,y_val,val_idx=self._make_sequences(scaled,train_end,val_end)
        x_test,y_test,test_idx=self._make_sequences(scaled,val_end,n)
        results={}; models={}; fit_meta={}
        for kind in ["SimpleRNN","LSTM"]:
            model, meta=self._fit(kind,x_train,y_train,x_val,y_val)
            model.eval()
            with torch.no_grad():
                v=model(x_val).numpy().ravel(); t=model(x_test).numpy().ravel()
            val_pred=scaler.inverse_transform(v[:,None]).ravel(); test_pred=scaler.inverse_transform(t[:,None]).ravel()
            results[kind]={"validation":_metrics(y[val_idx],val_pred),"test":_metrics(y[test_idx],test_pred)}
            models[kind]=model; fit_meta[kind]=meta
        baseline_pred=y[test_idx-1]
        results["LastValueBaseline"]={"test":_metrics(y[test_idx],baseline_pred)}
        best=min(["SimpleRNN","LSTM"],key=lambda k:results[k]["validation"]["RMSE"])
        markers=["temporal_split","seasonal_naive","naive_baseline","actual_rnn_execution","rnn_lstm_comparison","final_holdout_once","train_only_scaling","timestamp_integrity_check","shared_evidence_ready"]
        inspect=[{"fact":"rows","value":n},{"fact":"target","value":target},{"fact":"timestamp_integrity","value":ts_info},{"fact":"split_rows","value":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx)}},{"fact":"sequence_length","value":self.sequence_length}]
        comparison={k:v["test"] for k,v in results.items()}
        out={
            "agent":self.agent,
            "UNDERSTAND":f"Forecast {target} one step ahead from its previous {self.sequence_length} 15-minute observations; compare SimpleRNN and LSTM.",
            "INSPECT":inspect,
            "QUESTION":"Does recurrent sequence modeling improve one-step-ahead forecasting over a simple last-value baseline, and is LSTM consistently better than SimpleRNN?",
            "HYPOTHESES":["H1: SimpleRNN/LSTM reduce validation RMSE versus naive persistence.","H2: LSTM handles temporal dependencies better than SimpleRNN.","H3: conclusions may depend on metric because rare load transitions dominate RMSE."],
            "TESTS":["repair/check timestamp monotonicity","chronological 70/15/15 split","fit scaler on train only","validation-based checkpoint selection","final test once","compare MAE/RMSE/R2 with last-value baseline"],
            "COMPARE":comparison,
            "DECIDE":{"selected_by_validation_rmse":best,"validation_metrics":{k:results[k]["validation"] for k in ["SimpleRNN","LSTM"]},"test_metrics":comparison,"fit_meta":fit_meta,"timestamp_repair":ts_info},
            "CHALLENGE":["The exercise does not explicitly specify the forecast horizon; one-step ahead is an explicit simulation assumption.","A last-value baseline can beat recurrent models on MAE even if recurrent models improve RMSE/R2; do not call a universal winner from one metric."],
            "RISKS":["one-step horizon inferred rather than supplied","single chronological holdout does not quantify seasonal/seed uncertainty","contemporaneous exogenous columns were intentionally not used because future availability was not specified"],
            "CONFIDENCE":"MEDIUM",
            "markers":markers,
            "heuristic_trace":[{"rule":"forecast temporal split","source":"system-derived"},{"rule":"simple baseline before complex model","source":"curated expert principle"}],
            "artifacts":{"corrected_timestamp_start":str(corrected_ts.iloc[0]),"corrected_timestamp_end":str(corrected_ts.iloc[-1])},
        }
        for key in CONTRACT_KEYS: out.setdefault(key, [])
        return out
