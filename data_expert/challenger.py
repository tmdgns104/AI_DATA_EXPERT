from __future__ import annotations
from typing import Any

class Challenger:
    def review(self,problem,task_spec,result):
        issues=[];countertests=[];outputs={o.get("agent"):o for o in result.get("expert_outputs",[])};ml=outputs.get("machine-learning",{});markers=set(ml.get("markers",[]));problem_type=task_spec.get("problem_type");intent=task_spec.get("intent",{})
        def issue(code,severity,message,evidence=None):issues.append({"code":code,"severity":severity,"message":message,"evidence":evidence})
        if result.get("expert_errors"):issue("EXPERT_ERROR","CRITICAL","One or more selected experts failed during execution.",result["expert_errors"])
        modality=problem.get("profile",{}).get("modality","tabular")
        if modality=="tabular" and task_spec.get("target") and "TRAIN_MODEL" in intent.get("intents",[]):
            if "dummy_baseline" not in markers and problem_type in {"classification","regression"}:issue("NO_SIMPLE_BASELINE","CRITICAL","Supervised decision lacks a dummy/simple baseline.",sorted(markers))
            if "final_holdout_once" not in markers and problem_type in {"classification","regression"}:issue("TEST_SELECTION_RISK","CRITICAL","Final holdout isolation is not evidenced.",sorted(markers))
            if task_spec.get("prediction_time")=="UNKNOWN":issue("PREDICTION_TIME_UNKNOWN","WARNING","Prediction time is unknown, so feature availability/leakage cannot be fully certified.")
            if task_spec.get("business_cost")=="UNKNOWN":issue("BUSINESS_COST_UNKNOWN","WARNING","Business error costs are unknown; metric/threshold optimality may not match operations.")
        if problem_type=="classification" and modality=="tabular":
            if "imbalance_metric" not in markers:issue("CLASS_IMBALANCE_METRIC","CRITICAL","Classification result lacks imbalance-aware metrics.")
            countertests.append("Inspect per-class recall/precision and threshold tradeoffs, not only aggregate accuracy.")
        if problem_type=="regression":countertests.append("Inspect residuals and worst-performing segments; overall RMSE may hide local failure.")
        if task_spec.get("group_id") and "group" not in task_spec.get("split_strategy","").lower():issue("GROUP_SPLIT","CRITICAL","Repeated entity detected but TaskSpec does not require a group-aware split.",task_spec.get("group_id"))
        if task_spec.get("timestamp") and intent.get("primary_intent")=="FORECAST" and "chronological" not in task_spec.get("split_strategy","").lower():issue("TIME_SPLIT","CRITICAL","Forecasting requires chronological/backtest validation.")
        if task_spec.get("causal_or_predictive")=="causal":
            if "prediction_vs_causality" not in markers:issue("CAUSAL_OVERCLAIM","CRITICAL","Causal request lacks evidence separating prediction from causal identification.")
            countertests.append("Check treatment assignment, baseline balance, temporal ordering, and unmeasured confounding assumptions.")
        if "DL_TRAIN" in intent.get("intents",[]):
            dl=outputs.get("deep-learning",{});dlm=set(dl.get("markers",[]))
            if "actual_torch_training" not in dlm:issue("DL_NOT_EXECUTED","WARNING","Deep-learning intent did not produce actual PyTorch training evidence.",sorted(dlm))
            if "small_batch_overfit" not in dlm:issue("NO_SMALL_BATCH_SANITY","WARNING","Deep-learning path lacks a small-batch overfit sanity check.")
        if result.get("domain_context",{}).get("status")=="FOUND":countertests.append("Verify retrieved domain constraints against the actual company/process source before deployment.")
        else:issue("NO_DOMAIN_CONTEXT","INFO","No domain knowledge was retrieved; statistical validity does not imply process validity.")
        critical=[x for x in issues if x["severity"]=="CRITICAL"];warnings=[x for x in issues if x["severity"]=="WARNING"];status="FAIL" if critical else ("REVIEW" if warnings else "PASS");return {"status":status,"issues":issues,"countertests":countertests,"summary":f"{len(critical)} critical, {len(warnings)} warning, {len(issues)-len(critical)-len(warnings)} info issues"}
