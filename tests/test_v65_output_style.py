import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data_expert'))
from output_renderer_v65 import HumanFriendlyRendererV65

class V65OutputStyleTests(unittest.TestCase):
    def test_final_notebook_style_contract(self):
        c=HumanFriendlyRendererV65().style_contract()
        for key in [
            'observe_before_labeling_problem',
            'anomaly_then_evidence_then_fix',
            'eda_must_support_next_decision',
            'mark_arbitrary_choices_as_assumptions',
            'validation_selects_test_reports',
            'compare_metric_tradeoffs',
            'model_complexity_claims_must_be_scoped',
            'hide_internal_agent_terms',
            'comments_explain_why_not_what',
            'avoid_ai_report_tone',
        ]:
            self.assertTrue(c[key])

    def test_solver_contains_natural_reasoning_guards(self):
        text=(ROOT/'.agents/skills/ai-data-expert/scripts/solve_timeseries_rnn_v5.py').read_text(encoding='utf-8')
        required=[
            '아직 시계열이라고 단정하지 않고',
            "autocorr(lag=x)",
            '최적이라고 정한 건 아니고',
            '24시간 패턴까지 직접 입력으로 쓰려면 96개',
            '파라미터 수가 더 많아서 완전히 같은 복잡도의 모델 비교라고 보기는 어려움',
            "test_table['MAE'].idxmin()",
            "test_table['RMSE'].idxmin()",
            'SimpleRNN과 LSTM의 RMSE 차이',
        ]
        for marker in required:
            self.assertIn(marker,text)
        for banned in ['DataGuard','Argument Ledger','final status: REVIEW','rolling-origin backtest']:
            self.assertNotIn(banned,text)

if __name__=='__main__': unittest.main()
