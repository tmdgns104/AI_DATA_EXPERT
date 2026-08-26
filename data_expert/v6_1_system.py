from __future__ import annotations
from pathlib import Path
from v6_system import V6System
from task_spec_v61 import TaskSpecBuilderV61
from challenger_v61 import ChallengerV61
from modality_verifier_v61 import ModalityVerifierV61

class V61System(V6System):
    def __init__(self,root:str|Path|None=None):
        super().__init__(root=root); self.spec_builder=TaskSpecBuilderV61(); self.challenger=ChallengerV61(); self.verifier=ModalityVerifierV61()

EnhancedSystem=V61System
