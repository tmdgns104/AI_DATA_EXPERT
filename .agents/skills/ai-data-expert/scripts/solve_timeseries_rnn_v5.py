from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
import nbformat


def md(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip())


def build(input_path: Path, data_path: Path, output_path: Path, target: str, timestamp: str, sequence_length: int = 32):
    input_path = Path(input_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    original = nbformat.read(input_path, as_version=4)
    cells = [c.copy() for c in original.cells]

    cells += [
        md(f"""
        ## 풀이 방향

        먼저 데이터를 확인하고 시간 순서가 실제로 있는지 본 다음 SimpleRNN과 LSTM을 비교하기로 했음.

        과제에 예측 시점과 입력 길이가 따로 정해져 있지 않아서 이번 실습에서는 **최근 {sequence_length}개 값({sequence_length * 15 // 60}시간)으로 다음 1개 값(15분 뒤)**을 예측하는 조건으로 정했음.
        이 값이 최적이라고 정한 건 아니고, 데이터에서 시간 패턴을 확인한 뒤 하루 전 같은 시각을 쓰는 기준 모델도 같이 비교함.
        """),
        md("## 1. 데이터 불러오기"),
        code(f"""
        from pathlib import Path
        import copy, random, urllib.request, warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from IPython.display import Markdown, display

        warnings.filterwarnings('ignore')
        RANDOM_STATE = 42
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

        FILE_NAME = {data_path.name!r}
        candidates = [Path(FILE_NAME), Path('examples') / FILE_NAME, Path('..') / 'examples' / FILE_NAME]
        DATA_PATH = next((p for p in candidates if p.exists()), candidates[0])
        DATA_URL = 'https://drive.usercontent.google.com/download?id=1ijmTFmw9YeDxfthAEUblJB5dgBjEG-Se&export=download&confirm=t'
        if not DATA_PATH.exists():
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)

        df = pd.read_csv(DATA_PATH)
        print('shape:', df.shape)
        display(df.head())
        quality = pd.DataFrame({{'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'unique': df.nunique(dropna=False)}})
        display(quality)
        """),
        md("""
        데이터 크기와 결측치를 확인한 뒤 `date`가 실제 시간 순서대로 이어지는지 확인해봄.
        아직 시계열이라고 단정하지 않고 날짜 변환 결과와 `NSM` 흐름을 같이 봤음.
        """),
        code(f"""
        TARGET = {target!r}
        TIMESTAMP = {timestamp!r}
        raw_time = pd.to_datetime(df[TIMESTAMP], format='%d/%m/%Y %H:%M', errors='coerce')
        raw_delta = raw_time.diff()
        bad_order = raw_delta[raw_delta < pd.Timedelta(0)]
        print('날짜 변환 실패:', int(raw_time.isna().sum()))
        print('시간이 뒤로 가는 행:', len(bad_order))
        if len(bad_order):
            pos = bad_order.index[0]
            display(df.loc[max(0, pos-2):pos+2, [TIMESTAMP, 'NSM', TARGET]])
        """),
        md("""
        확인해보니 하루 마지막 부분에서 같은 날짜의 `00:00`이 나와 시간이 뒤로 가는 것처럼 보였음.
        그런데 `NSM`은 23시대 값 다음에 0으로 다시 시작해서 실제로는 다음 날 자정으로 보는 게 자연스러웠음.
        그래서 `NSM`이 23시대에서 0으로 바뀌는 행만 다음 날짜로 보정하고 다시 확인함.
        """),
        code("""
        nsm = pd.to_numeric(df['NSM'], errors='coerce').fillna(-1).to_numpy()
        midnight_fix = np.zeros(len(df), dtype=bool)
        midnight_fix[1:] = (nsm[1:] == 0) & (nsm[:-1] >= 23 * 3600)
        df['timestamp'] = raw_time + pd.to_timedelta(midnight_fix.astype(int), unit='D')

        deltas = df['timestamp'].diff().dropna()
        print('보정한 자정 행:', int(midnight_fix.sum()))
        print('중복 시간:', int(df['timestamp'].duplicated().sum()))
        print('시간순 정렬:', bool(df['timestamp'].is_monotonic_increasing))
        print('가장 많은 시간 간격:', deltas.mode().iloc[0])
        print('15분이 아닌 간격:', int((deltas != pd.Timedelta(minutes=15)).sum()))

        assert df['timestamp'].notna().all()
        assert df['timestamp'].is_monotonic_increasing
        assert not df['timestamp'].duplicated().any()
        assert (deltas == pd.Timedelta(minutes=15)).all()
        """),
        md("""
        보정 후에는 중복 없이 15분 간격으로 계속 이어졌음. 그래서 이 데이터는 시간 순서를 유지해서 분석하는 게 맞다고 판단했음.

        다른 변수도 있지만 이번 과제는 SimpleRNN과 LSTM의 차이를 같은 조건에서 비교하는 게 목적이라 입력은 과거 `Usage_kWh` 하나로 맞췄음.
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].hist(df[TARGET], bins=40)
        axes[0].set(title='Usage_kWh 분포', xlabel='Usage_kWh', ylabel='count')
        axes[1].plot(df['timestamp'].iloc[:96*7], df[TARGET].iloc[:96*7], linewidth=1)
        axes[1].set(title='처음 1주일 흐름', xlabel='time', ylabel='Usage_kWh')
        plt.tight_layout()
        plt.show()

        lag_table = pd.DataFrame({
            'lag': [1, 32, 96],
            '간격': ['15분 전', '8시간 전', '24시간 전'],
            '상관계수': [df[TARGET].autocorr(lag=x) for x in [1, 32, 96]],
        })
        display(lag_table.round(4))
        """),
        md(f"""
        직전 시점과의 관계를 먼저 확인하고, 24시간 전 값과도 관계가 있는지 같이 봤음.
        그래서 **직전값 기준**과 **하루 전 같은 시각 기준**을 둘 다 넣어서 RNN/LSTM이 실제로 더 나은지 비교하기로 했음.

        입력 길이 {sequence_length}개는 과제에서 정해진 값이 아니라 최근 {sequence_length * 15 // 60}시간 흐름을 보도록 이번 실습에서 정한 조건임. 24시간 패턴까지 직접 입력으로 쓰려면 96개 같은 다른 길이도 따로 비교해야 함.
        """),
        md("## 2. 시계열 데이터 만들기"),
        code(f"""
        y = df[TARGET].astype(float).to_numpy()
        n = len(y)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        scaler = StandardScaler()
        scaler.fit(y[:train_end].reshape(-1, 1))
        y_scaled = scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        SEQ_LEN = {sequence_length}
        def make_sequences(start, end):
            idx = np.arange(max(start, SEQ_LEN), end)
            X = np.stack([y_scaled[i-SEQ_LEN:i] for i in idx]).astype(np.float32)[:, :, None]
            Y = y_scaled[idx].astype(np.float32)[:, None]
            return torch.from_numpy(X), torch.from_numpy(Y), idx

        X_train, y_train, idx_train = make_sequences(0, train_end)
        X_val, y_val, idx_val = make_sequences(train_end, val_end)
        X_test, y_test, idx_test = make_sequences(val_end, n)

        split_table = pd.DataFrame({{
            'split': ['Train', 'Validation', 'Test'],
            'rows': [len(idx_train), len(idx_val), len(idx_test)],
            'start': [df['timestamp'].iloc[idx_train[0]], df['timestamp'].iloc[idx_val[0]], df['timestamp'].iloc[idx_test[0]]],
            'end': [df['timestamp'].iloc[idx_train[-1]], df['timestamp'].iloc[idx_val[-1]], df['timestamp'].iloc[idx_test[-1]]],
        }})
        display(split_table)
        """),
        md("""
        시계열이라 행을 섞지 않고 앞쪽 70%를 Train, 다음 15%를 Validation, 마지막 15%를 Test로 사용했음.
        Scaling도 전체 데이터에 먼저 맞추지 않고 Train 구간에만 맞춰 뒤쪽 정보가 미리 들어가지 않게 했음.
        """),
        md("## 3. 기준 모델"),
        code("""
        def metrics(y_true, y_pred):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
                'R2': r2_score(y_true, y_pred),
            }

        seasonal_lag = 96
        persistence_val = y[idx_val-1]
        seasonal_val = y[idx_val-seasonal_lag]
        baseline_val_table = pd.DataFrame([
            {'Model': 'Persistence', **metrics(y[idx_val], persistence_val)},
            {'Model': 'SeasonalNaive', **metrics(y[idx_val], seasonal_val)},
        ]).set_index('Model')
        display(baseline_val_table.round(4))
        """),
        md("""
        복잡한 모델끼리만 비교하면 실제로 좋아진 건지 알기 어려워서 단순한 두 기준을 먼저 잡았음.
        직전값은 lag 1 관계를, 하루 전 같은 시각은 lag 96 관계를 이용한 기준임.
        """),
        md("## 4. SimpleRNN / LSTM 학습"),
        code("""
        class SequenceRegressor(nn.Module):
            def __init__(self, kind, hidden=24):
                super().__init__()
                recurrent = nn.RNN if kind == 'SimpleRNN' else nn.LSTM
                self.rnn = recurrent(input_size=1, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])

        def fit_model(kind, epochs=5):
            random.seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)
            model = SequenceRegressor(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            loss_fn = nn.MSELoss()
            loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=False)
            best_state, best_val, best_epoch, history = None, np.inf, 0, []
            for epoch in range(1, epochs+1):
                model.train(); losses = []
                for xb, yb in loader:
                    optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step(); losses.append(loss.item())
                model.eval()
                with torch.no_grad(): val_loss = loss_fn(model(X_val), y_val).item()
                history.append([epoch, float(np.mean(losses)), float(val_loss)])
                if val_loss < best_val:
                    best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state); model.eval()
            return model, pd.DataFrame(history, columns=['epoch','train_loss','val_loss']), best_epoch

        simple_rnn, rnn_history, rnn_best_epoch = fit_model('SimpleRNN')
        lstm, lstm_history, lstm_best_epoch = fit_model('LSTM')

        def parameter_count(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_table = pd.DataFrame({
            'Model': ['SimpleRNN', 'LSTM'],
            'Parameters': [parameter_count(simple_rnn), parameter_count(lstm)],
        }).set_index('Model')
        display(param_table)
        """),
        md("""
        두 모델은 입력 데이터, hidden 크기, optimizer, learning rate, batch size, epoch 수를 같게 두고 비교했음.
        다만 LSTM은 gate 구조 때문에 같은 hidden 크기여도 파라미터 수가 더 많아서 완전히 같은 복잡도의 모델 비교라고 보기는 어려움.
        """),
        md("## 5. Validation 결과 비교"),
        code("""
        def predict_original(model, X):
            model.eval()
            with torch.no_grad(): pred_scaled = model(X).numpy().ravel()
            return scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()

        rnn_val_pred = predict_original(simple_rnn, X_val)
        lstm_val_pred = predict_original(lstm, X_val)
        validation_table = pd.DataFrame([
            {'Model':'Persistence', **metrics(y[idx_val], persistence_val)},
            {'Model':'SeasonalNaive', **metrics(y[idx_val], seasonal_val)},
            {'Model':'SimpleRNN', **metrics(y[idx_val], rnn_val_pred)},
            {'Model':'LSTM', **metrics(y[idx_val], lstm_val_pred)},
        ]).set_index('Model')
        display(validation_table.round(4))
        selected_model = validation_table['RMSE'].idxmin()
        print('Validation RMSE 기준 선택:', selected_model)
        """),
        md("""
        모델이나 기준 방법 중 어떤 걸 최종 후보로 볼지는 Validation RMSE로 정했음.
        Test 결과가 좋아 보인다는 이유로 여기서 정한 선택을 바꾸지는 않음.
        """),
        md("## 6. Test 결과 비교"),
        code("""
        persistence_test = y[idx_test-1]
        seasonal_test = y[idx_test-seasonal_lag]
        rnn_test_pred = predict_original(simple_rnn, X_test)
        lstm_test_pred = predict_original(lstm, X_test)
        test_table = pd.DataFrame([
            {'Model':'Persistence', **metrics(y[idx_test], persistence_test)},
            {'Model':'SeasonalNaive', **metrics(y[idx_test], seasonal_test)},
            {'Model':'SimpleRNN', **metrics(y[idx_test], rnn_test_pred)},
            {'Model':'LSTM', **metrics(y[idx_test], lstm_test_pred)},
        ]).set_index('Model')
        display(test_table.round(4))

        main_model = selected_model
        base_rmse = test_table.loc['Persistence','RMSE']
        selected_rmse = test_table.loc[main_model,'RMSE']
        base_gain = (base_rmse-selected_rmse)/base_rmse*100
        rnn_lstm_gap = abs(test_table.loc['SimpleRNN','RMSE']-test_table.loc['LSTM','RMSE']) / test_table.loc['SimpleRNN','RMSE']*100
        best_mae = test_table['MAE'].idxmin()
        best_rmse = test_table['RMSE'].idxmin()
        best_r2 = test_table['R2'].idxmax()

        change_word = '낮았음' if base_gain >= 0 else '높았음'
        result_text = (
            f'Validation에서 선택한 방법은 **{main_model}**였음.\n\n'
            f'최종 Test에서는 RMSE가 가장 낮은 방법이 **{best_rmse}**, MAE가 가장 낮은 방법이 **{best_mae}**, '
            f'R²가 가장 높은 방법이 **{best_r2}**였음.\n\n'
            f'선택된 방법의 Test RMSE는 **{selected_rmse:.4f}**이고 직전값 기준보다 **{base_gain:.2f}%** {change_word}. '
            f'SimpleRNN과 LSTM의 RMSE 차이는 약 **{rnn_lstm_gap:.2f}%**라서 차이가 작으면 압도적이라고 보기는 어려움.'
        )
        display(Markdown(result_text))
        """),
        md("## 7. 실제값과 예측값 비교"),
        code("""
        view = 300
        ts = df['timestamp'].iloc[idx_test[-view:]]
        plt.figure(figsize=(14,5))
        plt.plot(ts, y[idx_test[-view:]], label='Actual', linewidth=1.5)
        plt.plot(ts, rnn_test_pred[-view:], label='SimpleRNN', alpha=.85)
        plt.plot(ts, lstm_test_pred[-view:], label='LSTM', alpha=.85)
        plt.title('Usage_kWh 실제값과 예측값')
        plt.ylabel('Usage_kWh'); plt.xlabel('time'); plt.legend(); plt.tight_layout(); plt.show()
        """),
        md("## 8. 결론"),
        code("""
        selected_rmse = test_table.loc[selected_model, 'RMSE']
        selected_mae = test_table.loc[selected_model, 'MAE']
        selected_r2 = test_table.loc[selected_model, 'R2']
        p_rmse = test_table.loc['Persistence','RMSE']
        gain = (p_rmse-selected_rmse)/p_rmse*100
        gap = abs(test_table.loc['SimpleRNN','RMSE']-test_table.loc['LSTM','RMSE'])/test_table.loc['SimpleRNN','RMSE']*100
        conclusion = (
            f'이번 조건에서는 Validation RMSE 기준으로 **{selected_model}**을 선택했음.\n\n'
            f'- {selected_model} Test RMSE: **{selected_rmse:.4f}**\n'
            f'- {selected_model} Test MAE: **{selected_mae:.4f}**\n'
            f'- {selected_model} Test R²: **{selected_r2:.4f}**\n'
            f'- 직전값 기준 대비 RMSE 변화: **{gain:.2f}%**\n'
            f'- SimpleRNN과 LSTM RMSE 차이: **{gap:.2f}%**\n\n'
            f'RMSE, MAE, R²가 모두 같은 모델을 가리키지 않으면 그 차이도 같이 보는 게 맞음. '
            f'또 이번 결과는 최근 {SEQ_LEN}개 값으로 15분 뒤를 예측하도록 정한 조건에서 나온 결과라 입력 길이나 예측 시점을 바꾸면 다시 확인해야 함. '
            f'여러 기간으로 나눠 반복해서 확인하면 결과가 한 구간에만 우연히 맞은 건지도 더 볼 수 있음.'
        )
        display(Markdown(conclusion))
        """),
    ]

    notebook = nbformat.v4.new_notebook(metadata=original.metadata.copy())
    notebook.cells = cells
    notebook.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--target', default='Usage_kWh')
    ap.add_argument('--timestamp', default='date')
    ap.add_argument('--sequence-length', type=int, default=32)
    a = ap.parse_args()
    build(Path(a.input), Path(a.data), Path(a.output), a.target, a.timestamp, a.sequence_length)


if __name__ == '__main__':
    main()
