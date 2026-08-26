from __future__ import annotations
from typing import Any
from pathlib import Path
import pandas as pd

from hypothesis_engine import HypothesisEngine as V3HypothesisEngine, ExperimentManager as V3ExperimentManager
from data_guard_v4 import analyze_dataframe


class HypothesisEngineV4(V3HypothesisEngine):
    def generate(self, problem: dict[str, Any], task_spec: dict[str, Any], domain_context: dict[str, Any]):
        hs=super().generate(problem,task_spec,domain_context)
        guard=task_spec.get("data_guard") or {}
        ids=guard.get("drop_feature_columns",[])
        if guard.get("target_missing_count",0):
            hs.insert(0,{"id":"H-LABEL-STATE","statement":"Missing target values are unlabeled prediction rows, not a third class.","test_plan":"Separate labeled/unlabeled rows before any split/model fit and predict unlabeled only after evaluation.","falsified_if":"all target rows are labeled","priority":"HIGH"})
        if ids:
            hs.insert(0,{"id":"H-ID-PROXY","statement":"High-cardinality identifier/order features may create memorization leakage.","test_plan":f"Exclude {ids} and compare honest split performance.","falsified_if":"identifier has domain-justified predictive availability and survives entity/time holdout","priority":"HIGH"})
        if guard.get("group_strategy"):
            hs.insert(0,{"id":"H-RUN-SPLIT","statement":"Adjacent rows from the same entity/production run may leak across random split.","test_plan":f"Use {guard['group_strategy']} and verify zero group overlap.","falsified_if":"deployment only predicts future rows of already-seen groups under identical regime","priority":"HIGH"})
        if domain_context.get("facts"):
            hs.append({"id":"H-DOM-FACT","statement":"Retrieved source-bounded operational constraints can change feature eligibility, split, metric, or decision threshold.","test_plan":"Apply structured domain facts to TaskSpec and expert inputs; trace every applied fact to source.","falsified_if":"retrieved facts do not apply to this task","priority":"HIGH"})
        return hs


class ExperimentManagerV4(V3ExperimentManager):
    def run(self, problem: dict[str, Any], task_spec: dict[str, Any], hypotheses):
        result=super().run(problem,task_spec,hypotheses)
        path=problem.get("data_path"); target=task_spec.get("target")
        if path and Path(path).exists() and Path(path).suffix.lower()==".csv":
            try:
                df=pd.read_csv(path); guard=task_spec.get("data_guard") or analyze_dataframe(df,target)
                ev=result.setdefault("evidence",[])
                ev.append({"test":"target_label_state","value":{"missing":guard.get("target_missing_count",0),"labeled":guard.get("labeled_count"),"unlabeled":guard.get("unlabeled_count")}})
                ev.append({"test":"identifier_proxy_guard","value":guard.get("identifier_proxies",[])})
                ev.append({"test":"group_split_candidate","value":guard.get("group_strategy")})
                if target and target in df.columns and task_spec.get("problem_type")=="classification":
                    labeled=df[target].dropna(); counts=labeled.value_counts()
                    ev.append({"test":"class_balance_labeled_only","value":{str(k):int(v) for k,v in counts.items()},"minority_ratio":float(counts.min()/counts.sum()) if len(counts) else None})
            except Exception as exc:
                result.setdefault("warnings",[]).append(f"V4 experiment guard failed: {type(exc).__name__}: {exc}")
        return result
