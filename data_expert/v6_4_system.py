from __future__ import annotations
from pathlib import Path
from v6_3_system import V63System
from domain_rag_v64 import HybridDomainRAGV64

class V64System(V63System):
    def __init__(self,root:str|Path|None=None):
        super().__init__(root=root); self.rag=HybridDomainRAGV64(self.root/'domain_knowledge')
EnhancedSystem=V64System
