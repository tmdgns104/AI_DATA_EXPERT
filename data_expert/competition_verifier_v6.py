from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class ScoreCard:
    total:int; components:dict; critical_errors:list; status:str
    def to_dict(self): return asdict(self)

class CompetitionVerifierV6:
    WEIGHTS={'problem_understanding':10,'data_leakage_guard':15,'correct_validation':15,'metric_understanding':10,
             'baseline':10,'modeling':10,'failure_analysis':10,'verifier_challenger':10,'submission':5,'human_output':5}
    def score(self,evidence:dict)->ScoreCard:
        comp={k:(w if evidence.get(k,False) else 0) for k,w in self.WEIGHTS.items()}
        critical=[]
        for k in ['leakage_miss','test_reuse','wrong_metric','wrong_rag_evidence','expert_contradiction','notebook_failure','unsupported_causal_claim','silent_unknown_assumption']:
            if evidence.get(k,False): critical.append(k)
        total=sum(comp.values())
        status='FAIL' if any(x in critical for x in ['leakage_miss','test_reuse','wrong_metric','wrong_rag_evidence']) else ('REVIEW' if critical or total<90 else 'PASS')
        return ScoreCard(total=total,components=comp,critical_errors=critical,status=status)
