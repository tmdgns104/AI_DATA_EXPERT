from __future__ import annotations
from typing import Any
from domain_rag_v62 import HybridDomainRAGV62


class HybridDomainRAGV63(HybridDomainRAGV62):
    """Reject bundled demo evidence unless the active task is explicitly in scope."""

    def retrieve(self, problem: dict[str, Any], top_k: int = 4) -> dict[str, Any]:
        raw = super().retrieve(problem, top_k=max(top_k * 2, 8))
        p = problem.get('profile', {})
        target = self._norm(p.get('target'))
        domain = self._norm(p.get('domain'))
        process = self._norm(p.get('process'))
        task = self._norm(problem.get('task'))
        manufacturing_scope = target == 'yield percentage' or any(x in f'{task} {domain} {process}' for x in ['manufacturing','factory','yield','공정','제조','수율'])
        accepted=[]; rejected=list(raw.get('rejected_matches',[])); accepted_sources=set()
        for m in raw.get('matches',[]):
            source=str(m.get('source','')).lower().replace('\\','/')
            bundled = source.endswith('/domain_knowledge/manufacturing_example.md') or source.endswith('/domain_knowledge/manufacturing_constraints.json') or 'manufacturing_example.md' in source or 'manufacturing_constraints.json' in source
            if bundled and not manufacturing_scope:
                rejected.append({'source':m.get('source'),'reason':'bundled_manufacturing_demo_out_of_scope'}); continue
            accepted.append(m); accepted_sources.add(str(m.get('source','')))
            if len(accepted)>=top_k: break
        facts=[]; rejected_facts=list(raw.get('rejected_facts',[]))
        for f in raw.get('facts',[]):
            source=str(f.get('source','')).lower().replace('\\','/')
            bundled='manufacturing_constraints.json' in source
            if bundled and not manufacturing_scope:
                rejected_facts.append({'source':f.get('source'),'type':f.get('type'),'reason':'bundled_manufacturing_fact_out_of_scope'}); continue
            if accepted_sources and str(f.get('source','')) not in accepted_sources:
                rejected_facts.append({'source':f.get('source'),'type':f.get('type'),'reason':'fact_source_not_accepted'}); continue
            facts.append(f)
        raw['matches']=accepted; raw['facts']=facts; raw['rejected_matches']=rejected; raw['rejected_facts']=rejected_facts
        raw['status']='FOUND' if accepted or facts else 'NO_MATCH'
        raw['evidence_gate']={'version':'V6.3','accepted_matches':len(accepted),'rejected_matches':len(rejected),'accepted_facts':len(facts),'rejected_facts':len(rejected_facts),'manufacturing_scope':manufacturing_scope}
        return raw
