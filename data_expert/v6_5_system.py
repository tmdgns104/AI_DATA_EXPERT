from __future__ import annotations
from pathlib import Path
from v6_4_system import V64System
from competition_planner_v65 import CompetitionPlannerV65
from output_renderer_v65 import HumanFriendlyRendererV65

class V65System(V64System):
    def __init__(self, root: str | Path | None = None):
        super().__init__(root=root)
        self.competition_planner = CompetitionPlannerV65()
        self.renderer = HumanFriendlyRendererV65()

EnhancedSystem = V65System
