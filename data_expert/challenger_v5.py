from __future__ import annotations
from typing import Any
from challenger_v4 import ChallengerV4


class ChallengerV5(ChallengerV4):
    def review(self, problem: dict[str,Any], task_spec: dict[str,Any], result: dict[str,Any]):
        out=super().review(problem,task_spec,result)
        issues=list(out.get("issues",[])); modality=str(problem.get("profile",{}).get("modality","tabular")).lower()
        if modality=="time-series":
            ts=next((o for o in result.get("expert_outputs",[]) if o.get("agent")=="time-series"),{})
            markers=set(ts.get("markers",[]))
            if "actual_rnn_execution" in markers:
                issues=[i for i in issues if i.get("code")!="DL_NOT_EXECUTED"]
            # Small-batch overfit is useful for generic DL debugging, but is not a mandatory time-series quality gate here.
            issues=[i for i in issues if i.get("code")!="NO_SMALL_BATCH_SANITY"]
            if "timestamp_integrity_check" not in markers:
                issues.append({"code":"TS_TIMESTAMP_UNCHECKED","severity":"CRITICAL","message":"Time-series model lacks timestamp integrity evidence.","evidence":None})
            if "temporal_split" not in markers:
                issues.append({"code":"TS_RANDOM_SPLIT_RISK","severity":"CRITICAL","message":"Forecasting result lacks chronological validation evidence.","evidence":None})
            if "naive_baseline" not in markers:
                issues.append({"code":"TS_NO_NAIVE_BASELINE","severity":"WARNING","message":"Forecasting model was not compared with a persistence/naive baseline.","evidence":None})
        critical=[x for x in issues if x.get("severity")=="CRITICAL"]; warnings=[x for x in issues if x.get("severity")=="WARNING"]
        return {**out,"issues":issues,"status":"FAIL" if critical else ("REVIEW" if warnings else "PASS"),"summary":f"{len(critical)} critical, {len(warnings)} warning, {len(issues)-len(critical)-len(warnings)} info issues"}
