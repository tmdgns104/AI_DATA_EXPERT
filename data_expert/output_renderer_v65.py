from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RenderReview:
    score: int
    checks: dict
    notes: list

class HumanFriendlyRendererV65:
    """Keep reasoning rigorous internally, but render a natural notebook externally."""
    MAX_VISIBLE_SECTIONS = 10
    INTERNAL_TERMS = (
        'DataGuard', 'Argument Ledger', 'Evidence ID', 'Verifier',
        'final status: REVIEW', 'rolling-origin backtest'
    )

    def style_contract(self):
        return {
            'preserve_original_cells': True,
            'observe_before_labeling_problem': True,
            'anomaly_then_evidence_then_fix': True,
            'eda_must_support_next_decision': True,
            'mark_arbitrary_choices_as_assumptions': True,
            'validation_selects_test_reports': True,
            'compare_metric_tradeoffs': True,
            'model_complexity_claims_must_be_scoped': True,
            'hide_internal_agent_terms': True,
            'short_markdown': True,
            'comments_explain_why_not_what': True,
            'avoid_ai_report_tone': True,
            'allow_concise_korean_endings': True,
        }

    def review_text(self, markdown_sections: list[str], code_comments: list[str], internal_terms: list[str] | None = None) -> RenderReview:
        internal_terms = tuple(internal_terms or ()) + self.INTERNAL_TERMS
        joined='\n'.join(markdown_sections)
        checks = {
            'section_count': len(markdown_sections) <= self.MAX_VISIBLE_SECTIONS,
            'concise_markdown': all(len(x.split()) <= 220 for x in markdown_sections),
            'no_internal_terms': not any(t in joined for t in internal_terms),
            'comment_density': len(code_comments) <= 40,
            'validation_test_roles': ('Validation' not in joined or 'Test' not in joined or joined.find('Validation') <= joined.rfind('Test')),
        }
        notes=[]
        if not checks['no_internal_terms']:
            notes.append('내부 검증 용어를 사용자 Notebook 표현으로 바꿀 것')
        return RenderReview(score=round(100*sum(checks.values())/len(checks)), checks=checks, notes=notes)
