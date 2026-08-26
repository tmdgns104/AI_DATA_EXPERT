from __future__ import annotations
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_expert'))
from v6_4_system import V64System
class V64RagScope(unittest.TestCase):
 def test_negated_manufacturing_scope_is_not_activated(self):
  for task in ['external problem; no manufacturing context','this is not a factory task','non-manufacturing retail model','without manufacturing assumptions']:
   r=V64System(ROOT).rag.retrieve({'task':task,'profile':{'target':'power','modality':'tabular'}})
   self.assertEqual(r['status'],'NO_MATCH',task); self.assertFalse(r['evidence_gate']['manufacturing_scope'],task)
 def test_explicit_manufacturing_still_retrieves(self):
  r=V64System(ROOT).rag.retrieve({'task':'predict manufacturing yield before process completion','profile':{'target':'target','modality':'tabular'}})
  self.assertEqual(r['status'],'FOUND'); self.assertTrue(r['evidence_gate']['manufacturing_scope'])
if __name__=='__main__':unittest.main()
