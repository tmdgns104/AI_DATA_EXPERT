from __future__ import annotations
from pathlib import Path
from typing import Any
from v5_system import V5System
from competition_spec_v6 import CompetitionSpecBuilderV6
from competition_verifier_v6 import CompetitionVerifierV6
from output_renderer_v6 import HumanFriendlyRendererV6
from competition_planner_v6 import CompetitionPlannerV6
from shared_evidence_v5 import SharedEvidenceStore
from argument_ledger_v5 import ArgumentLedger

class V6System(V5System):
    def __init__(self,root:str|Path|None=None):
        super().__init__(root=root)
        self.competition_builder=CompetitionSpecBuilderV6(); self.competition_verifier=CompetitionVerifierV6(); self.renderer=HumanFriendlyRendererV6(); self.competition_planner=CompetitionPlannerV6()
    def prepare_competition(self,raw:dict[str,Any],df=None):
        spec=self.competition_builder.build(raw); plan=self.competition_planner.inspect(spec,df); store=SharedEvidenceStore(); ledger=ArgumentLedger()
        store.publish_many({'competition.spec':spec.to_dict(),'competition.metric':spec.metric,'competition.validation':plan.get('inferred_validation'),'competition.guard':plan},'competition-supervisor',confidence='HIGH')
        ledger.add(id='H-METRIC-001',question='Which metric must control model selection?',hypotheses=['Use a generic metric','Use the competition metric exactly'],required_evidence=['competition specification'],observations=[spec.metric],counterarguments=['Local diagnostics may use extra metrics but cannot replace the competition metric.'],decision=f'Use {spec.metric} as the selection metric.',status='SUPPORTED',confidence='HIGH',provenance=[spec.source_url],next_questions=[])
        ledger.add(id='H-VALID-001',question='Which validation scheme matches the data-generating process?',hypotheses=['Random split','Competition-aware split'],required_evidence=['category','risk flags','competition specification'],observations=[spec.validation,list(spec.risk_flags)],counterarguments=['A single split can still be noisy.'],decision=f'Use {spec.validation} validation and keep final test labels unavailable.',status='SUPPORTED',confidence='HIGH',provenance=[spec.source_url],next_questions=['Would repeated/rolling validation materially change model ranking?'])
        return {'competition_spec':spec.to_dict(),'competition_plan':plan,'shared_evidence':store.snapshot(),'argument_ledger':ledger.snapshot(),'output_style':self.renderer.style_contract()}

EnhancedSystem=V6System
