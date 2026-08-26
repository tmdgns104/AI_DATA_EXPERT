from __future__ import annotations

from pathlib import Path
from typing import Any

from v4_system import V4System, RuntimeVerifierV4
from task_spec_v5 import TaskSpecBuilderV5
from time_series_dl_v5 import TimeSeriesDLExpertV5
from shared_evidence_v5 import SharedEvidenceStore
from argument_ledger_v5 import ArgumentLedger
from domain_rag_v5 import HybridDomainRAGV5
from challenger_v5 import ChallengerV5


class RouterV5:
    def __init__(self, base): self.base=base
    def route(self, problem: dict[str,Any], task_spec: dict[str,Any]):
        route=self.base.route(problem,task_spec)
        modality=str(problem.get("profile",{}).get("modality","tabular")).lower()
        intents=set(task_spec.get("intent",{}).get("intents",[])); neg=set(task_spec.get("intent",{}).get("negated",[]))
        if modality=="time-series" and "FORECAST" in intents and "FORECAST" not in neg:
            # V5 treats sequence DL as a time-series specialist responsibility, avoiding contradictory TS/DL states.
            selected=[a for a in route["selected_agents"] if a not in {"deep-learning","data-analyst"}]
            if "time-series" not in selected:selected.append("time-series")
            route["selected_agents"]=selected; route["execution_order"]=["time-series"]
            route["primary_agent"]="time-series"
            route.setdefault("routing_reasons",{})["time-series"]=["chronological sequence forecasting / RNN-LSTM comparison"]
            route["routing_reasons"].pop("deep-learning",None); route["routing_reasons"].pop("data-analyst",None)
        return route


class ModalityVerifierV5:
    def __init__(self): self.base=RuntimeVerifierV4()
    def verify(self, problem: dict[str,Any], result: dict[str,Any]):
        base=self.base.verify(problem,result); checks=list(base.get("checks",[]))
        modality=str(problem.get("profile",{}).get("modality","tabular")).lower()
        if modality=="time-series":
            # Remove generic DL requirements that are handled by the dedicated time-series specialist.
            checks=[c for c in checks if c["name"] not in {"dl_routed","actual_dl_execution","dl_small_batch_sanity"}]
            ts=next((o for o in result.get("expert_outputs",[]) if o.get("agent")=="time-series"),{})
            markers=set(ts.get("markers",[])); decision=ts.get("DECIDE",{})
            def add(name,ok,evidence=None,severity="ERROR"):
                checks.append({"name":name,"pass":bool(ok),"severity":severity,"evidence":evidence})
            add("timeseries_specialist_routed",result.get("routing",{}).get("primary_agent")=="time-series",result.get("routing",{}).get("execution_order"))
            add("timestamp_integrity_checked","timestamp_integrity_check" in markers,decision.get("timestamp_repair"))
            add("chronological_split","temporal_split" in markers,sorted(markers))
            add("train_only_scaling","train_only_scaling" in markers,sorted(markers))
            add("naive_baseline_present","naive_baseline" in markers,sorted(markers))
            add("actual_rnn_execution","actual_rnn_execution" in markers,sorted(markers))
            add("rnn_lstm_compared","rnn_lstm_comparison" in markers,decision.get("test_metrics"))
            add("final_test_isolated","final_holdout_once" in markers,sorted(markers))
            ledger=result.get("argument_ledger",{})
            add("argument_ledger_present",bool(ledger.get("nodes")),ledger.get("nodes"))
            store=result.get("shared_evidence",{})
            add("shared_evidence_present",store.get("record_count",0)>0,store.get("record_count"))
            if "explicit forecast horizon" in result.get("task_spec",{}).get("unknowns",[]):
                add("forecast_horizon_confirmed",False,"one-step ahead inferred",severity="WARNING")
        errors=[c for c in checks if not c["pass"] and c["severity"]=="ERROR"]; warnings=[c for c in checks if not c["pass"] and c["severity"]=="WARNING"]
        return {"status":"FAIL" if errors else ("REVIEW" if warnings else "PASS"),"passed":sum(bool(c["pass"]) for c in checks),"total":len(checks),"score_pct":100*sum(bool(c["pass"]) for c in checks)/len(checks) if checks else 100.0,"errors":len(errors),"warnings":len(warnings),"checks":checks}


class V5System(V4System):
    def __init__(self, root: str|Path|None=None):
        super().__init__(root=root)
        self.spec_builder=TaskSpecBuilderV5(); self.router=RouterV5(self.router); self.verifier=ModalityVerifierV5()
        self.rag=HybridDomainRAGV5(self.root/"domain_knowledge"); self.challenger=ChallengerV5()
        self.engines["time-series"]=TimeSeriesDLExpertV5()

    def _build_arguments(self, problem, task_spec, result, store):
        ledger=ArgumentLedger(); ts=next((o for o in result.get("expert_outputs",[]) if o.get("agent")=="time-series"),{})
        d=ts.get("DECIDE",{}); tinfo=d.get("timestamp_repair",{}); metrics=d.get("test_metrics",{})
        ledger.add(id="H-TIME-001",question="Can the raw date column be used as-is as a chronological index?",hypotheses=["Raw timestamp is already chronological","Daily midnight rows are encoded ambiguously and require repair"],required_evidence=["row order","NSM reset","timestamp differences"],observations=[tinfo],counterarguments=["Automatic repair is dataset-specific and must retain provenance."],decision="Use NSM-reset midnight repair before chronological modeling." if tinfo.get("midnight_repairs",0)>0 else "No timestamp repair required.",status="SUPPORTED",confidence="HIGH" if tinfo.get("monotonic_after_repair") and not tinfo.get("irregular_interval_count") else "MEDIUM",provenance=["time-series expert:timestamp_integrity"],next_questions=["Does another authoritative timestamp definition exist in the domain documentation?"])
        ledger.add(id="H-SPLIT-001",question="Is random splitting acceptable for this forecasting exercise?",hypotheses=["Random split is acceptable","Chronological split is required to avoid future-to-past leakage"],required_evidence=["forecast intent","timestamp order"],observations=[task_spec.get("split_strategy")],counterarguments=["A single chronological holdout still does not measure all seasonal regimes."],decision="Use chronological Train/Validation/Test.",status="SUPPORTED",confidence="HIGH",provenance=["TaskSpecV5"],next_questions=["Should rolling-origin backtesting replace the single holdout in a production evaluation?"])
        if metrics:
            base=metrics.get("LastValueBaseline",{}); rnn=metrics.get("SimpleRNN",{}); lstm=metrics.get("LSTM",{})
            best=d.get("selected_by_validation_rmse")
            mae_winner=min([(k,v.get("MAE",1e99)) for k,v in metrics.items()],key=lambda x:x[1])[0]
            status="SUPPORTED" if best and best==mae_winner else "INCONCLUSIVE"
            ledger.add(id="H-MODEL-001",question="Does one recurrent model clearly dominate the alternatives?",hypotheses=["SimpleRNN is sufficient","LSTM is better","Persistence baseline remains competitive"],required_evidence=["validation RMSE","test RMSE","test MAE","baseline"],observations=[{"validation_selected":best,"test":metrics,"mae_winner":mae_winner}],counterarguments=["Winner changes by metric" if status=="INCONCLUSIVE" else "One holdout and one seed are not enough for deployment certainty."],decision=f"Validation RMSE selects {best}; test MAE winner is {mae_winner}." if best else "No defensible winner.",status=status,confidence="MEDIUM",provenance=["time-series expert:model comparison"],next_questions=["Repeat across rolling-origin folds and multiple seeds before deployment."])
        ledger.add(id="H-HORIZON-001",question="What forecast horizon does the exercise require?",hypotheses=["One-step 15-minute ahead","Longer forecasting horizon"],required_evidence=["assignment text","business requirement"],observations=[task_spec.get("prediction_time")],counterarguments=["The assignment only says prediction; it does not explicitly define a horizon."],decision="Use one-step ahead only for this simulation.",status="INCONCLUSIVE",confidence="LOW",provenance=["assignment text"],next_questions=["Confirm required horizon with the instructor/spec before production use."])
        return ledger.snapshot()

    def run(self, problem: dict[str,Any]):
        # Reimplement V4 execution so the shared store can be populated before challenger/verifier.
        task_spec=self.spec_builder.build(problem)
        enriched={**problem,"profile":dict(problem.get("profile",{}))}; enriched["task_spec"]=task_spec; enriched["data_guard"]=task_spec.get("data_guard")
        domain_context=self.rag.retrieve(enriched)
        from task_spec_v4 import apply_domain_facts
        task_spec,applied=apply_domain_facts(task_spec,domain_context.get("facts",[])); enriched["task_spec"]=task_spec; enriched["domain_context"]=domain_context; enriched["domain_facts"]=domain_context.get("facts",[])
        enriched["profile"]["retrieved_domain_text"]="\n\n".join(m.get("text","") for m in domain_context.get("matches",[])[:4]); enriched["profile"]["domain_facts"]=domain_context.get("facts",[])
        route=self.router.route(enriched,task_spec); outs=[];errors=[]; store=SharedEvidenceStore()
        store.publish_many({"task_spec":task_spec,"routing":route,"domain_backend":domain_context.get("retrieval_backend")},"supervisor",confidence="HIGH")
        for agent in route["execution_order"]:
            engine=self.engines.get(agent)
            if engine is None:continue
            try:
                out=engine.run(enriched); out["TASK_SPEC"]={k:task_spec.get(k) for k in ["observation_unit","target","prediction_time","problem_type","split_strategy","primary_metric","business_cost","causal_or_predictive"]}; out["DOMAIN_EVIDENCE"]=[{"source":m.get("source"),"chunk_id":m.get("chunk_id"),"rank":m.get("rank"),"score":m.get("score"),"text":m.get("text","")[:800]} for m in domain_context.get("matches",[])[:4]]; out["DOMAIN_FACTS_APPLIED"]=applied; outs.append(out)
                store.publish(f"expert.{agent}.decision",out.get("DECIDE"),agent,confidence=out.get("CONFIDENCE","MEDIUM")); store.publish(f"expert.{agent}.markers",out.get("markers",[]),agent,confidence="HIGH")
            except Exception as exc: errors.append({"agent":agent,"type":type(exc).__name__,"error":str(exc)}); store.publish(f"expert.{agent}.error",str(exc),agent,confidence="HIGH",status="ERROR")
        hypotheses=self.hypothesis_engine.generate(enriched,task_spec,domain_context); experiments=self.experiment_manager.run(enriched,task_spec,hypotheses)
        result={"problem_id":problem.get("id"),"task_spec":task_spec,"data_guard":task_spec.get("data_guard"),"domain_context":domain_context,"routing":route,"expert_outputs":outs,"expert_errors":errors,"hypotheses":hypotheses,"experiment_evidence":experiments,"shared_evidence":store.snapshot()}
        result["argument_ledger"]=self._build_arguments(enriched,task_spec,result,store)
        result["challenger"]=self.challenger.review(enriched,task_spec,result)
        result["verification"]=self.verifier.verify(enriched,result)
        return result


EnhancedSystem=V5System
