from __future__ import annotations
from typing import Any

from challenger import Challenger as V3Challenger


class ChallengerV4(V3Challenger):
    def review(self, problem: dict[str,Any], task_spec: dict[str,Any], result: dict[str,Any]):
        base=super().review(problem,task_spec,result)
        issues=list(base.get("issues",[])); counter=list(base.get("countertests",[])); guard=task_spec.get("data_guard") or {}; outputs={o.get("agent"):o for o in result.get("expert_outputs",[])}; ml=outputs.get("machine-learning",{}); markers=set(ml.get("markers",[]))
        if "TRAIN_MODEL" in set(task_spec.get("intent",{}).get("negated",[])):
            issues=[i for i in issues if i.get("code") not in {"DL_NOT_EXECUTED","NO_SMALL_BATCH_SANITY"}]
        def add(code,severity,message,evidence=None):
            if code not in {i.get("code") for i in issues}:issues.append({"code":code,"severity":severity,"message":message,"evidence":evidence})
        tabular_supervised = problem.get("profile",{}).get("modality","tabular") == "tabular" and task_spec.get("problem_type") in {"classification","regression"}
        if guard.get("target_missing_count",0) and "target_missing_separated" not in markers and tabular_supervised:
            add("TARGET_MISSING_AS_CLASS","CRITICAL","Target-missing rows are not evidenced as separated unlabeled predictions.",guard.get("target_missing_count"))
        if guard.get("drop_feature_columns") and "identifier_proxy_excluded" not in markers and tabular_supervised:
            add("IDENTIFIER_PROXY_INPUT","CRITICAL","High-cardinality ID/order proxy was detected but exclusion is not evidenced.",guard.get("drop_feature_columns"))
        if guard.get("group_strategy") and "group_aware_split" not in markers and tabular_supervised:
            add("HONEST_GROUP_SPLIT","CRITICAL","Production/entity group structure was detected but model evidence lacks group-aware split.",guard.get("group_strategy"))
        if "group_aware_split" in markers and "group_overlap_zero" not in markers:
            add("GROUP_OVERLAP","CRITICAL","Group-aware split is claimed without zero-overlap evidence.")
        if task_spec.get("domain_facts_applied") and "domain_context_injected" not in markers and tabular_supervised:
            add("DOMAIN_FACT_NOT_APPLIED","WARNING","Applicable structured domain facts were retrieved but model evidence does not show context injection.",task_spec.get("domain_facts_applied"))
        per_class=ml.get("DECIDE",{}).get("per_class",{})
        if per_class:
            low={k:v for k,v in per_class.items() if v.get("recall",1)>=0 and v.get("recall",1)<.60}
            if low:add("LOW_CLASS_RECALL","WARNING","At least one class recall is below 0.60; aggregate metrics are not sufficient for operational approval.",low)
        counter += ["Compare random-row score against entity/run/time-isolated score when sequence structure exists.","Check that target-missing rows are predicted after evaluation rather than included as a class."]
        critical=[x for x in issues if x["severity"]=="CRITICAL"]; warnings=[x for x in issues if x["severity"]=="WARNING"]
        status="FAIL" if critical else ("REVIEW" if warnings else "PASS")
        return {"status":status,"issues":issues,"countertests":counter,"summary":f"{len(critical)} critical, {len(warnings)} warning, {len(issues)-len(critical)-len(warnings)} info issues"}
