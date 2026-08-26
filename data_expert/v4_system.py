from __future__ import annotations

from pathlib import Path
from typing import Any

import enhanced_system_v2 as legacy
from v3_system import IntentRouterV3, RuntimeVerifierV3, CONTRACT_KEYS
from task_spec_v4 import TaskSpecBuilderV4, apply_domain_facts
from domain_rag_v4 import HybridDomainRAG
from hypothesis_engine_v4 import HypothesisEngineV4, ExperimentManagerV4
from challenger_v4 import ChallengerV4
from advanced_ml_v4 import AdvancedMLExpertV4
from dl_engine_v4 import TorchDeepLearningExpertV4


class RuntimeVerifierV4:
    def __init__(self): self.base=RuntimeVerifierV3()

    def verify(self, problem: dict[str,Any], result: dict[str,Any]):
        base=self.base.verify(problem,result)
        spec=result["task_spec"]; negated=set(spec.get("intent",{}).get("negated",[])); checks=list(base.get("checks",[]))
        if "TRAIN_MODEL" in negated:
            checks=[c for c in checks if c["name"] not in {"dl_routed","actual_dl_execution","dl_small_batch_sanity","supervised_ml_routed"}]
        outputs={o.get("agent"):o for o in result.get("expert_outputs",[])}; ml=outputs.get("machine-learning",{}); markers=set(ml.get("markers",[])); guard=spec.get("data_guard") or {}
        def add(name,ok,evidence=None,severity="ERROR"):
            checks.append({"name":name,"pass":bool(ok),"severity":severity,"evidence":evidence})
        tabular_supervised = problem.get("profile",{}).get("modality","tabular") == "tabular" and spec.get("problem_type") in {"regression","classification"}
        if tabular_supervised and spec.get("target") and "TRAIN_MODEL" in set(spec.get("intent",{}).get("intents",[])) and "TRAIN_MODEL" not in negated:
            if guard.get("target_missing_count",0)>0:add("target_missing_not_class","target_missing_separated" in markers,guard.get("target_missing_count"))
            if guard.get("drop_feature_columns"):add("identifier_proxy_excluded","identifier_proxy_excluded" in markers,guard.get("drop_feature_columns"))
            if guard.get("group_strategy"):
                add("group_aware_split","group_aware_split" in markers,guard.get("group_strategy"))
                add("group_overlap_zero","group_overlap_zero" in markers,ml.get("DECIDE",{}).get("group_overlap"))
            if result.get("task_spec",{}).get("domain_facts_applied"):
                add("domain_fact_injected","domain_context_injected" in markers,result.get("task_spec",{}).get("domain_facts_applied"))
        add("hybrid_rag_trace",bool(result.get("domain_context",{}).get("retrieval_backend")),result.get("domain_context",{}).get("retrieval_backend"),severity="WARNING")
        errors=[c for c in checks if not c["pass"] and c["severity"]=="ERROR"]; warnings=[c for c in checks if not c["pass"] and c["severity"]=="WARNING"]
        passed=sum(1 for c in checks if c["pass"]); total=len(checks); status="FAIL" if errors else ("REVIEW" if warnings else "PASS")
        return {"status":status,"passed":passed,"total":total,"score_pct":100*passed/total if total else 100.0,"errors":len(errors),"warnings":len(warnings),"checks":checks}


class V4System:
    def __init__(self, root: str|Path|None=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        self.spec_builder=TaskSpecBuilderV4(); self.router=IntentRouterV3(); self.rag=HybridDomainRAG(self.root/"domain_knowledge")
        self.hypothesis_engine=HypothesisEngineV4(); self.experiment_manager=ExperimentManagerV4(); self.challenger=ChallengerV4(); self.verifier=RuntimeVerifierV4()
        self.engines={
            "data-analyst":legacy.EnhancedDataAnalyst(),
            "machine-learning":AdvancedMLExpertV4(),
            "deep-learning":TorchDeepLearningExpertV4(),
            "vision-ai":legacy.EnhancedVision(),
            "time-series":legacy.EnhancedTS(),
            "big-data":legacy.EnhancedBigData(),
            "mlops":legacy.EnhancedMLOps(),
        }

    def run(self, problem: dict[str,Any]):
        task_spec=self.spec_builder.build(problem)
        enriched={**problem,"profile":dict(problem.get("profile",{}))}; enriched["task_spec"]=task_spec; enriched["data_guard"]=task_spec.get("data_guard")
        domain_context=self.rag.retrieve(enriched)
        task_spec,applied=apply_domain_facts(task_spec,domain_context.get("facts",[])); enriched["task_spec"]=task_spec; enriched["domain_context"]=domain_context; enriched["domain_facts"]=domain_context.get("facts",[])
        enriched["profile"]["retrieved_domain_text"]="\n\n".join(m.get("text","") for m in domain_context.get("matches",[])[:4])
        enriched["profile"]["domain_facts"] = domain_context.get("facts",[])
        route=self.router.route(enriched,task_spec)
        outs=[];errors=[]
        for agent in route["execution_order"]:
            engine=self.engines.get(agent)
            if engine is None:continue
            try:
                out=engine.run(enriched)
                out["TASK_SPEC"]={k:task_spec.get(k) for k in ["observation_unit","target","prediction_time","problem_type","split_strategy","primary_metric","business_cost","causal_or_predictive"]}
                out["DOMAIN_EVIDENCE"]=[{"source":m.get("source"),"chunk_id":m.get("chunk_id"),"rank":m.get("rank"),"score":m.get("score"),"bm25_score":m.get("bm25_score"),"vector_score":m.get("vector_score"),"text":m.get("text","")[:1000]} for m in domain_context.get("matches",[])[:4]]
                out["DOMAIN_FACTS_APPLIED"]=applied
                outs.append(out)
            except Exception as exc:
                errors.append({"agent":agent,"type":type(exc).__name__,"error":str(exc)})
        hypotheses=self.hypothesis_engine.generate(enriched,task_spec,domain_context); experiments=self.experiment_manager.run(enriched,task_spec,hypotheses)
        result={"problem_id":problem.get("id"),"task_spec":task_spec,"data_guard":task_spec.get("data_guard"),"domain_context":domain_context,"routing":route,"expert_outputs":outs,"expert_errors":errors,"hypotheses":hypotheses,"experiment_evidence":experiments}
        result["challenger"]=self.challenger.review(enriched,task_spec,result); result["verification"]=self.verifier.verify(enriched,result); return result


EnhancedSystem=V4System

def find_agent(result,name):return next((x for x in result.get("expert_outputs",[]) if x.get("agent")==name),None)
