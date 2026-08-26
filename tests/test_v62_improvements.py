from __future__ import annotations

import sys, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_expert"))
from v6_2_system import V62System
from competition_spec_v6 import CompetitionSpecBuilderV6
from metric_adapters_v62 import weighted_pinball, wrmsse


class V62Improvements(unittest.TestCase):
    def test_01_time_group_compound_validation(self):
        n = 240
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "device_id": np.repeat(np.arange(24), 10),
            "x": np.arange(n),
            "isFraud": np.r_[np.zeros(228), np.ones(12)],
        })
        raw = {"category":"classification","slug":"ieee-fraud-detection","name":"IEEE-CIS Fraud Detection","metric":"roc_auc","direction":"max","target":"isFraud","validation":"time-aware","submission_columns":["isFraud"],"risk_flags":["imbalance","time_order","entity_leakage"],"source_url":"https://www.kaggle.com/competitions/ieee-fraud-detection"}
        plan = V62System(ROOT).prepare_competition(raw, df)["competition_plan"]
        self.assertEqual(plan["inferred_validation"], "temporal-group")
        self.assertEqual(set(plan["validation_components"]), {"chronological", "group-isolation"})

    def test_02_wrong_domain_demo_and_facts_rejected(self):
        result = V62System(ROOT).rag.retrieve({"task":"forecast steel electricity usage","profile":{"target":"Usage_kWh","modality":"time-series"}})
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertEqual(result["facts"], [])
        self.assertGreaterEqual(result["evidence_gate"]["rejected_matches"], 1)

    def test_03_baselines_participate_in_champion_selection(self):
        data = ROOT / "examples" / "Steel_industry_data.csv"
        r = V62System(ROOT).run({"task":"forecast and compare SimpleRNN and LSTM","data_path":str(data),"profile":{"target":"Usage_kWh","modality":"time-series","timestamp":"date"}})
        d = r["expert_outputs"][0]["DECIDE"]
        self.assertIn("Persistence", d["validation_metrics"])
        self.assertIn("SeasonalNaive", d["validation_metrics"])
        self.assertIn(d["selected_by_validation_rmse"], d["validation_metrics"])
        self.assertIn("baseline_in_champion_selection", r["expert_outputs"][0]["markers"])

    def test_04_adaptive_argument_policy_emits_next_experiment(self):
        data = ROOT / "examples" / "Steel_industry_data.csv"
        r = V62System(ROOT).run({"task":"compare SimpleRNN and LSTM predictions","data_path":str(data),"profile":{"target":"Usage_kWh","modality":"time-series","timestamp":"date"}})
        ids = {x["id"] for x in r["adaptive_next_experiments"]}
        self.assertIn("EXP-HORIZON-BLOCKED-001", ids)

    def test_05_metric_adapters(self):
        y = np.array([10., 20.])
        pred = np.array([[8.,10.,12.],[18.,20.,22.]])
        self.assertGreaterEqual(weighted_pinball(y,pred,[.1,.5,.9]), 0.0)
        yt=np.array([[1.,2.],[3.,4.]])
        yp=np.array([[1.,1.],[2.,5.]])
        self.assertGreaterEqual(wrmsse(yt,yp,[1.,2.],[1.,2.]), 0.0)

if __name__ == "__main__": unittest.main()
