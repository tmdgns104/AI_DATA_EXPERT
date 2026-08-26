from __future__ import annotations

from pathlib import Path
from typing import Any

import enhanced_system_v2 as legacy
from task_spec import TaskSpecBuilder
from domain_rag import DomainRAG
from hypothesis_engine import HypothesisEngine, ExperimentManager
from challenger import Challenger
from dl_engine_v3 import TorchDeepLearningExpert
from advanced_ml import AdvancedMLExpert

CONTRACT_KEYS = ["UNDERSTAND","INSPECT","QUESTION","HYPOTHESES","TESTS","COMPARE","DECIDE","CHALLENGE","RISKS","CONFIDENCE"]

class IntentRouterV3:
    def route(self, problem: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
        p=problem.get("profile",{}); modality=str(p.get("modality","tabular")).lower(); idec=task_spec["intent"]; intents=set(idec.get("intents",[])); negated=set(idec.get("negated",[])); primary_intent=idec.get("primary_intent","ANALYZE_ONLY"); selected=[]; reasons={}
        def add(agent,reason):
            if agent not in selected:selected.append(agent)
            reasons.setdefault(agent,[]).append(reason)
        def remove(agent):
            if agent in selected:selected.remove(agent)
            reasons.pop(agent,None)
        scale=float(p.get("size_gb",0) or 0)>=10 or int(p.get("rows",0) or 0)>=10_000_000 or bool(p.get("streaming")) or modality=="big-data"
        if scale or intents & {"ARCHITECTURE","REALTIME_PIPELINE"}:add("big-data","scale/streaming architecture constraint")
        if modality in {"image","vision","video"}:
            add("vision-ai","vision modality framing/data quality")
            if "TRAIN_MODEL" in intents or "VISION_TRAIN" in intents or "DL_TRAIN" in intents:add("deep-learning","vision training intent")
        elif modality=="time-series":
            add("data-analyst","time-series data quality/structure")
            if "FORECAST" in intents and "FORECAST" not in negated:add("time-series","forecast/backtest intent")
        elif modality=="tabular":
            if primary_intent in {"ANALYZE_ONLY","AUDIT_DATA","CAUSAL_ANALYSIS"} or task_spec.get("target"):add("data-analyst","tabular observation/data-quality review")
            if ("TRAIN_MODEL" in intents or "COMPARE_MODELS" in intents) and "TRAIN_MODEL" not in negated and task_spec.get("target"):add("machine-learning","supervised modeling intent")
        if "SURVIVAL_ANALYSIS" in intents or task_spec.get("problem_type")=="survival":add("data-analyst","time/event/censoring structure");add("machine-learning","censor-aware survival analysis")
        if "DL_TRAIN" in intents and "TRAIN_MODEL" not in negated:add("deep-learning","explicit deep-learning execution")
        if "MONITOR_EXISTING_MODEL" in intents or p.get("monitoring"):add("mlops","production monitoring/root-cause intent")
        if "DEPLOY_MODEL" in intents or p.get("deployment"):add("mlops","deployment/serving constraints")
        if "CAUSAL_ANALYSIS" in intents:
            add("data-analyst","causal framing/confounding review")
            if task_spec.get("target") and "TRAIN_MODEL" in intents:add("machine-learning","predictive model separated from causal claim")
        if "TRAIN_MODEL" in negated:
            remove("deep-learning")
            if task_spec.get("problem_type")!="survival" and primary_intent not in {"MONITOR_EXISTING_MODEL","DEPLOY_MODEL"}:remove("machine-learning")
        if "FORECAST" in negated:remove("time-series")
        if not selected:add("data-analyst","default evidence-first analysis")
        priority={"big-data":10,"data-analyst":20,"vision-ai":30,"time-series":35,"machine-learning":40,"deep-learning":50,"mlops":60};order=sorted(selected,key=lambda a:priority[a])
        if primary_intent=="MONITOR_EXISTING_MODEL" and "mlops" in selected:primary="mlops"
        elif primary_intent in {"ARCHITECTURE","REALTIME_PIPELINE"} and "big-data" in selected:primary="big-data"
        elif primary_intent=="FORECAST" and "time-series" in selected:primary="time-series"
        elif modality in {"image","vision","video"} and "vision-ai" in selected:primary="vision-ai"
        elif task_spec.get("problem_type")=="survival" and "machine-learning" in selected:primary="machine-learning"
        elif primary_intent=="DL_TRAIN" and "deep-learning" in selected:primary="deep-learning"
        elif "machine-learning" in selected:primary="machine-learning"
        else:primary=order[0]
        return {"primary_agent":primary,"selected_agents":selected,"execution_order":order,"routing_reasons":reasons,"intent":idec,"semantic_flags":{"problem_type":task_spec.get("problem_type"),"causal_or_predictive":task_spec.get("causal_or_predictive"),"group_id":task_spec.get("group_id"),"timestamp":task_spec.get("timestamp")}}

class RuntimeVerifierV3:
    def verify(self, problem: dict[str,Any], result: dict[str,Any]) -> dict[str,Any]:
        checks=[]
        def add(name,ok,evidence=None,severity="ERROR"):checks.append({"name":name,"pass":bool(ok),"severity":severity,"evidence":evidence})
        spec=result["task_spec"];route=result["routing"];selected=set(route["selected_agents"]);outputs={o["agent"]:o for o in result.get("expert_outputs",[])};add("no_expert_execution_error",not result.get("expert_errors"),result.get("expert_errors"))
        for agent in selected:
            o=outputs.get(agent);add(f"{agent}:output_present",o is not None)
            if o:
                add(f"{agent}:reasoning_contract_complete",all(bool(o.get(k)) for k in CONTRACT_KEYS),{k:bool(o.get(k)) for k in CONTRACT_KEYS});add(f"{agent}:decision_present",bool(o.get("DECIDE")));add(f"{agent}:heuristic_trace",bool(o.get("heuristic_trace")),severity="WARNING")
        add("taskspec_problem_type",bool(spec.get("problem_type")),spec.get("problem_type"));add("taskspec_observation_unit",bool(spec.get("observation_unit")),spec.get("observation_unit"));add("hypothesis_engine_ran",len(result.get("hypotheses",[]))>=1,len(result.get("hypotheses",[])));add("experiment_manager_ran",result.get("experiment_evidence",{}).get("status") in {"PASS","NO_TABULAR_DATA"},result.get("experiment_evidence",{}).get("status"));challenger=result.get("challenger",{});add("challenger_ran",bool(challenger),challenger.get("status"));add("challenger_no_critical",challenger.get("status")!="FAIL",challenger.get("issues"));
        if challenger.get("status")=="REVIEW":add("challenger_no_warning",False,challenger.get("issues"),severity="WARNING")
        intent=set(spec.get("intent",{}).get("intents",[]));p=problem.get("profile",{});pt=spec.get("problem_type")
        if p.get("modality","tabular")=="tabular" and spec.get("target") and "TRAIN_MODEL" in intent and pt in {"regression","classification"}:
            add("supervised_ml_routed","machine-learning" in selected,route["execution_order"]);ml=outputs.get("machine-learning",{});markers=set(ml.get("markers",[]));add("simple_baseline","dummy_baseline" in markers,sorted(markers));add("final_test_isolated","final_holdout_once" in markers,sorted(markers))
            if pt=="classification":add("classification_metric","imbalance_metric" in markers,sorted(markers));add("probability_quality","probability_quality" in markers,sorted(markers));add("per_class_metrics","per_class_metrics" in markers,sorted(markers));add("threshold_validation","threshold_validation" in markers,sorted(markers))
            if pt=="regression":add("uncertainty_diagnostic","uncertainty_diagnostic" in markers,sorted(markers));add("segment_failure_analysis","segment_failure_analysis" in markers,sorted(markers));add("ood_diagnostic","ood_diagnostic" in markers,sorted(markers))
            if spec.get("prediction_time")=="UNKNOWN":add("prediction_time_known",False,"UNKNOWN",severity="WARNING")
            if spec.get("business_cost")=="UNKNOWN":add("business_cost_known",False,"UNKNOWN",severity="WARNING")
        if pt=="survival":ml=outputs.get("machine-learning",{});add("survival_censor_aware","survival_censoring" in set(ml.get("markers",[])),ml.get("markers"))
        negated=set(spec.get("intent",{}).get("negated",[]))
        if "FORECAST" in intent and "FORECAST" not in negated:
            ts=outputs.get("time-series",{});ms=set(ts.get("markers",[]));add("forecast_routed","time-series" in selected,route["execution_order"]);add("forecast_naive_baseline","seasonal_naive" in ms,sorted(ms));add("forecast_temporal_split","temporal_split" in ms,sorted(ms))
        if "DL_TRAIN" in intent or (p.get("modality") in {"image","vision","video"} and "TRAIN_MODEL" in intent):
            add("dl_routed","deep-learning" in selected,route["execution_order"]);dl=outputs.get("deep-learning",{});ms=set(dl.get("markers",[]));executable=bool(p.get("image_npz")) or (p.get("modality","tabular")=="tabular" and p.get("target"));add("actual_dl_execution",("actual_torch_training" in ms) if executable else True,sorted(ms),severity="ERROR" if executable else "WARNING");add("dl_small_batch_sanity",("small_batch_overfit" in ms) if executable else True,sorted(ms),severity="ERROR" if executable else "WARNING")
        if "MONITOR_EXISTING_MODEL" in intent:add("mlops_routed","mlops" in selected,route["execution_order"])
        if "ARCHITECTURE" in intent or "REALTIME_PIPELINE" in intent:add("bigdata_routed","big-data" in selected,route["execution_order"])
        if "CAUSAL_ANALYSIS" in intent:ml=outputs.get("machine-learning",{});markers=set(ml.get("markers",[]));add("causal_guard","prediction_vs_causality" in markers or "machine-learning" not in selected,sorted(markers))
        if p.get("deployment"):add("deployment_prediction_time",spec.get("prediction_time")!="UNKNOWN",spec.get("prediction_time"));add("deployment_business_cost",spec.get("business_cost")!="UNKNOWN",spec.get("business_cost"),severity="WARNING")
        errors=[c for c in checks if not c["pass"] and c["severity"]=="ERROR"];warnings=[c for c in checks if not c["pass"] and c["severity"]=="WARNING"];status="FAIL" if errors else ("REVIEW" if warnings else "PASS");passed=sum(1 for c in checks if c["pass"]);total=len(checks);return {"status":status,"passed":passed,"total":total,"score_pct":100*passed/total if total else 100.0,"errors":len(errors),"warnings":len(warnings),"checks":checks}

class V3System:
    def __init__(self, root: str|Path|None=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1];self.spec_builder=TaskSpecBuilder();self.router=IntentRouterV3();self.rag=DomainRAG(self.root/"domain_knowledge");self.hypothesis_engine=HypothesisEngine();self.experiment_manager=ExperimentManager();self.challenger=Challenger();self.verifier=RuntimeVerifierV3();self.engines={"data-analyst":legacy.EnhancedDataAnalyst(),"machine-learning":AdvancedMLExpert(),"deep-learning":TorchDeepLearningExpert(),"vision-ai":legacy.EnhancedVision(),"time-series":legacy.EnhancedTS(),"big-data":legacy.EnhancedBigData(),"mlops":legacy.EnhancedMLOps()}
    def run(self,problem):
        task_spec=self.spec_builder.build(problem);enriched={**problem,"profile":dict(problem.get("profile",{}))};enriched["task_spec"]=task_spec;domain_context=self.rag.retrieve(enriched);route=self.router.route(enriched,task_spec);outs=[];errors=[]
        for agent in route["execution_order"]:
            engine=self.engines.get(agent)
            if engine is None:continue
            try:
                out=engine.run(enriched);out["TASK_SPEC"]={k:task_spec[k] for k in ["observation_unit","target","prediction_time","problem_type","split_strategy","primary_metric","causal_or_predictive"]};out["DOMAIN_EVIDENCE"]=[{"source":m["source"],"chunk_id":m["chunk_id"],"score":m["score"]} for m in domain_context.get("matches",[])[:3]];outs.append(out)
            except Exception as exc:errors.append({"agent":agent,"type":type(exc).__name__,"error":str(exc)})
        hypotheses=self.hypothesis_engine.generate(enriched,task_spec,domain_context);experiments=self.experiment_manager.run(enriched,task_spec,hypotheses);result={"problem_id":problem.get("id"),"task_spec":task_spec,"domain_context":domain_context,"routing":route,"expert_outputs":outs,"expert_errors":errors,"hypotheses":hypotheses,"experiment_evidence":experiments};result["challenger"]=self.challenger.review(enriched,task_spec,result);result["verification"]=self.verifier.verify(enriched,result);return result

EnhancedSystem=V3System

def find_agent(result,name):return next((x for x in result.get("expert_outputs",[]) if x.get("agent")==name),None)
