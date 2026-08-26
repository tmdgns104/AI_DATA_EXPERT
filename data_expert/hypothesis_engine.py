from __future__ import annotations
from pathlib import Path
from typing import Any
import math
import numpy as np
import pandas as pd

class HypothesisEngine:
    def generate(self,problem,task_spec,domain_context):
        hypotheses=[]
        def add(hid,statement,test,falsified_if,priority="MEDIUM"):hypotheses.append({"id":hid,"statement":statement,"test_plan":test,"falsified_if":falsified_if,"priority":priority})
        add("H-DQ-1","Data quality or duplicated observations may distort evaluation.","Measure missingness, duplicate rate, target validity, and suspicious near-constant columns.","quality checks are clean and repeated results are stable","HIGH")
        if task_spec.get("target"):
            add("H-LEAK-1","At least one feature may contain target/post-outcome information.","Inspect prediction-time availability plus near-deterministic feature-target relationships.","no suspicious timing/proxy feature is found","HIGH");add("H-BASE-1","A simple baseline may be sufficient.","Compare a dummy and simple linear/logistic baseline against nonlinear candidates using train-only validation.","complex model materially and consistently beats simple baselines","HIGH")
        if task_spec.get("group_id"):add("H-GROUP-1","Random row split may overstate generalization because entities repeat.",f"Compare or require a group-aware split by {task_spec['group_id']}.","deployment is to repeated rows of already-seen entities","HIGH")
        if task_spec.get("timestamp") or task_spec.get("intent",{}).get("primary_intent")=="FORECAST":add("H-TIME-1","Random split may leak future regimes into training.","Use chronological or rolling validation and compare seasonal/naive baselines.","task is purely retrospective with no future generalization claim","HIGH")
        pt=task_spec.get("problem_type")
        if pt=="classification":add("H-IMB-1","Accuracy may hide minority-class failure.","Measure class balance, macro-F1, balanced accuracy and probability metrics; inspect threshold tradeoffs.","classes are balanced and all class-wise metrics are stable","HIGH");add("H-CAL-1","Raw probabilities may be poorly calibrated.","Compute Brier/log-loss/ECE when binary probabilities are available.","calibration error is small for the decision use-case")
        if pt=="regression":add("H-SEG-1","Overall RMSE may hide a segment where errors are much worse.","Measure error by segments.","segment errors are similar")
        if task_spec.get("causal_or_predictive")=="causal":add("H-CAUSAL-1","Predictive association may be confounded.","Check treatment assignment, balance, temporal ordering and causal assumptions.","valid identification design is documented","HIGH")
        if task_spec.get("intent",{}).get("primary_intent")=="MONITOR_EXISTING_MODEL":add("H-OPS-1","Drift may be pipeline/sensor change rather than model aging.","Compare schema, feature stats, labels, performance and deployment changes.","root cause isolates model aging","HIGH")
        if "DL_TRAIN" in task_spec.get("intent",{}).get("intents",[]):add("H-DL-1","Silent pipeline/label bug may look like optimization failure.","Run shape/label checks and small-batch overfit sanity.","small batch can be overfit","HIGH")
        if domain_context.get("status")=="FOUND":add("H-DOM-1","Domain constraints may invalidate a statistically attractive feature/model.","Cross-check retrieved domain notes.","no conflicting operational constraint","HIGH")
        return hypotheses

class ExperimentManager:
    def run(self,problem,task_spec,hypotheses):
        path=problem.get("data_path");evidence=[]
        if not path or not Path(path).exists() or Path(path).suffix.lower()!=".csv":return {"status":"NO_TABULAR_DATA","evidence":evidence}
        try:df=pd.read_csv(path)
        except Exception as exc:return {"status":"READ_FAILED","error":f"{type(exc).__name__}: {exc}","evidence":evidence}
        evidence.append({"test":"duplicate_rate","value":float(df.duplicated().mean())});evidence.append({"test":"missing_rate_max","value":float(df.isna().mean().max()) if len(df.columns) else 0.0});evidence.append({"test":"constant_columns","value":[str(c) for c in df.columns if df[c].nunique(dropna=False)<=1]});target=task_spec.get("target")
        if target and target in df.columns:
            y=df[target]
            if task_spec.get("problem_type")=="classification":counts=y.value_counts(dropna=False);evidence.append({"test":"class_balance","value":{str(k):int(v) for k,v in counts.items()},"minority_ratio":float(counts.min()/counts.sum()) if len(counts) else None})
            elif pd.api.types.is_numeric_dtype(y):desc=y.describe(percentiles=[.01,.05,.5,.95,.99]).to_dict();evidence.append({"test":"target_distribution","value":{str(k):float(v) for k,v in desc.items() if pd.notna(v)}})
        gid=task_spec.get("group_id")
        if gid and gid in df.columns:
            counts=df[gid].value_counts(dropna=False);evidence.append({"test":"repeated_entity_structure","group_id":gid,"entities":int(counts.size),"max_rows_per_entity":int(counts.max()),"median_rows_per_entity":float(counts.median())})
        timestamp=task_spec.get("timestamp")
        if timestamp and timestamp in df.columns:
            ts=pd.to_datetime(df[timestamp],errors="coerce");evidence.append({"test":"timestamp_quality","parseable_ratio":float(ts.notna().mean()),"monotonic":bool(ts.dropna().is_monotonic_increasing),"min":str(ts.min()) if ts.notna().any() else None,"max":str(ts.max()) if ts.notna().any() else None})
        return {"status":"PASS","evidence":evidence,"hypotheses_tested":[h["id"] for h in hypotheses]}
