from __future__ import annotations
from pathlib import Path
from v6_2_system import V62System
from domain_rag_v63 import HybridDomainRAGV63
from competition_planner_v63 import CompetitionPlannerV63

class V63System(V62System):
    def __init__(self, root:str|Path|None=None):
        super().__init__(root=root)
        self.rag=HybridDomainRAGV63(self.root/'domain_knowledge')
        self.competition_planner=CompetitionPlannerV63()

EnhancedSystem=V63System
