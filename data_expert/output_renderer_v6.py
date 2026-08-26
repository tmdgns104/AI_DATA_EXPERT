from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RenderReview:
    score:int; checks:dict; notes:list

class HumanFriendlyRendererV6:
    """Keep internal reasoning rich while keeping user notebooks concise."""
    MAX_VISIBLE_SECTIONS=8
    def style_contract(self):
        return {
          'preserve_original_cells':True,'max_visible_sections':8,'hide_internal_agent_ids':True,
          'short_markdown':True,'comments_explain_why_not_what':True,'result_interpretation_conclusion':True,
          'avoid_ai_report_tone':True,'allow_concise_korean_endings':True,
        }
    def review_text(self, markdown_sections:list[str], code_comments:list[str], internal_terms:list[str]|None=None)->RenderReview:
        internal_terms=internal_terms or []
        checks={
          'section_count':len(markdown_sections)<=self.MAX_VISIBLE_SECTIONS,
          'concise_markdown':all(len(x.split())<=180 for x in markdown_sections),
          'no_internal_ids':not any(any(t in x for t in internal_terms) for x in markdown_sections),
          'comment_density':len(code_comments)<=40,
        }
        return RenderReview(score=round(100*sum(checks.values())/len(checks)),checks=checks,notes=[])
