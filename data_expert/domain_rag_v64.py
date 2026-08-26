from __future__ import annotations
import re
from typing import Any
from domain_rag_v63 import HybridDomainRAGV63

NEG_PATTERNS=[
    r'\bno\s+manufacturing\b', r'\bnot\s+(?:a\s+)?manufacturing\b', r'\bnon[- ]manufacturing\b',
    r'\bwithout\s+manufacturing\b', r'\bunrelated\s+to\s+manufacturing\b', r'\bnot\s+(?:a\s+)?factory\b',
    r'\bnon[- ]factory\b', r'\bwithout\s+factory\b', r'제조\s*(?:아님|아니|무관)', r'공정\s*(?:아님|아니|무관)'
]
POS_TERMS=['manufacturing','factory','yield','process quality','공정','제조','수율']

class HybridDomainRAGV64(HybridDomainRAGV63):
    @staticmethod
    def _manufacturing_scope(problem:dict[str,Any], norm) -> tuple[bool,dict[str,Any]]:
        p=problem.get('profile',{})
        task=norm(problem.get('task')); target=norm(p.get('target')); domain=norm(p.get('domain')); process=norm(p.get('process'))
        combined=f'{task} {domain} {process}'
        neg=[pat for pat in NEG_PATTERNS if re.search(pat,combined,re.I)]
        explicit_domain=any(term in f'{domain} {process}' for term in POS_TERMS)
        explicit_target=target in {'yield percentage','yield','수율'}
        positive=any(term in combined for term in POS_TERMS)
        scope=(explicit_domain or explicit_target or positive) and not (neg and not explicit_domain and not explicit_target)
        return scope,{'positive_signal':positive,'explicit_domain':explicit_domain,'explicit_target':explicit_target,'negated_patterns':neg}

    def retrieve(self,problem:dict[str,Any],top_k:int=4)->dict[str,Any]:
        from domain_rag_v62 import HybridDomainRAGV62
        raw=HybridDomainRAGV62.retrieve(self,problem,top_k=max(top_k*2,8))
        scope,trace=self._manufacturing_scope(problem,self._norm)
        accepted=[]; rejected=list(raw.get('rejected_matches',[])); accepted_sources=set()
        for m in raw.get('matches',[]):
            source=str(m.get('source','')).lower().replace('\\','/')
            bundled=any(x in source for x in ['manufacturing_example.md','manufacturing_constraints.json'])
            if bundled and not scope:
                rejected.append({'source':m.get('source'),'reason':'bundled_manufacturing_demo_out_of_scope'}); continue
            accepted.append(m); accepted_sources.add(str(m.get('source','')))
            if len(accepted)>=top_k: break
        facts=[]; rejected_facts=list(raw.get('rejected_facts',[]))
        for f in raw.get('facts',[]):
            source=str(f.get('source','')).lower().replace('\\','/')
            bundled='manufacturing_constraints.json' in source
            if bundled and not scope:
                rejected_facts.append({'source':f.get('source'),'type':f.get('type'),'reason':'bundled_manufacturing_fact_out_of_scope'}); continue
            if accepted_sources and str(f.get('source','')) not in accepted_sources:
                rejected_facts.append({'source':f.get('source'),'type':f.get('type'),'reason':'fact_source_not_accepted'}); continue
            facts.append(f)
        raw['matches']=accepted; raw['facts']=facts; raw['rejected_matches']=rejected; raw['rejected_facts']=rejected_facts
        raw['status']='FOUND' if accepted or facts else 'NO_MATCH'
        raw['evidence_gate']={'version':'V6.4','accepted_matches':len(accepted),'rejected_matches':len(rejected),'accepted_facts':len(facts),'rejected_facts':len(rejected_facts),'manufacturing_scope':scope,'scope_trace':trace}
        return raw
