from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
import nbformat


def md(text: str): return nbformat.v4.new_markdown_cell(dedent(text).strip())
def code(text: str): return nbformat.v4.new_code_cell(dedent(text).strip())


def build(input_path: Path, data_path: Path, output_path: Path, target: str, timestamp: str, sequence_length: int = 32):
    original=nbformat.read(input_path,as_version=4)
    cells=[c.copy() for c in original.cells]
    cells += [
        md(f"""
        # 풀이

        이 문제는 행을 무작위로 섞는 일반 회귀가 아니라 **시간 순서가 있는 15분 간격 시계열 예측**으로 처리합니다.
        `SimpleRNN`과 `LSTM`을 같은 Train/Validation/Test 구간과 같은 입력 길이로 비교합니다.

        - Target: `{target}`
        - 입력 길이: 이전 {sequence_length}개 관측값 ({sequence_length*15//60}시간)
        - 예측: 다음 1개 관측값(15분 뒤) — **과제에 horizon이 없어 이번 실습에서 명시적으로 둔 가정**
        - Split: 시간순 70% / 15% / 15%
        - Scaling: Train 구간에만 fit
        - 기준모델: 직전 값을 그대로 다음 값으로 예측하는 Persistence baseline
        - 모델 선택 기준: Validation RMSE
        - 최종 Test: 모델/체크포인트 결정 후 마지막에 평가
        """),
        md("## 1. 데이터 로딩과 기본 확인"),
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
        from IPython.display import display

        warnings.filterwarnings('ignore')
        RANDOM_STATE=42
        random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
        torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

        DATA_PATH=Path(r'{data_path.name}')
        DATA_URL='https://drive.usercontent.google.com/download?id=1ijmTFmw9YeDxfthAEUblJB5dgBjEG-Se&export=download&confirm=t'
        if not DATA_PATH.exists():
            print('데이터 파일을 다운로드합니다.')
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        df=pd.read_csv(DATA_PATH)
        print('shape:',df.shape)
        display(df.head())
        display(df.describe(include='all').T)
        """),
        md("""
        ## 2. 시간축 DataGuard

        원본 데이터의 `date`는 하루 마지막 자정을 같은 날짜의 `00:00`으로 기록한 행이 있어 그대로 파싱하면 시간이 뒤로 갑니다.
        `NSM`이 23시대에서 0으로 reset되는 행만 다음 날짜 자정으로 보정하고, 보정 뒤 **15분 간격·중복 0·단조 증가**인지 확인합니다.
        이 보정은 데이터셋 구조에 근거한 규칙이며 일반 시계열 데이터에 무조건 적용하는 규칙은 아닙니다.
        """),
        code(f"""
        TARGET='{target}'
        TIMESTAMP='{timestamp}'
        raw_time=pd.to_datetime(df[TIMESTAMP], format='%d/%m/%Y %H:%M', errors='coerce')
        nsm=pd.to_numeric(df['NSM'],errors='coerce').fillna(-1).to_numpy()
        midnight_fix=np.zeros(len(df),dtype=bool)
        midnight_fix[1:]=(nsm[1:]==0)&(nsm[:-1]>=23*3600)
        df['timestamp_corrected']=raw_time + pd.to_timedelta(midnight_fix.astype(int),unit='D')

        deltas=df['timestamp_corrected'].diff().dropna()
        print('timestamp parse failures:',int(df['timestamp_corrected'].isna().sum()))
        print('midnight repaired rows:',int(midnight_fix.sum()))
        print('duplicate timestamps:',int(df['timestamp_corrected'].duplicated().sum()))
        print('monotonic:',bool(df['timestamp_corrected'].is_monotonic_increasing))
        print('dominant interval:',deltas.mode().iloc[0])
        assert df['timestamp_corrected'].notna().all()
        assert df['timestamp_corrected'].is_monotonic_increasing
        assert not df['timestamp_corrected'].duplicated().any()
        assert (deltas==pd.Timedelta(minutes=15)).all()
        """),
        md("## 3. 시계열 분리와 Train-only Scaling"),
        code(f"""
        y=df[TARGET].astype(float).to_numpy()
        n=len(y); train_end=int(n*0.70); val_end=int(n*0.85)
        scaler=StandardScaler().fit(y[:train_end].reshape(-1,1))
        y_scaled=scaler.transform(y.reshape(-1,1)).ravel().astype(np.float32)
        SEQ_LEN={sequence_length}

        def make_sequences(start,end):
            idx=np.arange(max(start,SEQ_LEN),end)
            X=np.stack([y_scaled[i-SEQ_LEN:i] for i in idx]).astype(np.float32)[:,:,None]
            Y=y_scaled[idx].astype(np.float32)[:,None]
            return torch.from_numpy(X),torch.from_numpy(Y),idx

        X_train,y_train,idx_train=make_sequences(0,train_end)
        X_val,y_val,idx_val=make_sequences(train_end,val_end)
        X_test,y_test,idx_test=make_sequences(val_end,n)

        split_table=pd.DataFrame({{
            'split':['Train','Validation','Test'],
            'rows':[len(idx_train),len(idx_val),len(idx_test)],
            'start':[df['timestamp_corrected'].iloc[idx_train[0]],df['timestamp_corrected'].iloc[idx_val[0]],df['timestamp_corrected'].iloc[idx_test[0]]],
            'end':[df['timestamp_corrected'].iloc[idx_train[-1]],df['timestamp_corrected'].iloc[idx_val[-1]],df['timestamp_corrected'].iloc[idx_test[-1]]],
        }})
        display(split_table)
        print('sequence shapes:',X_train.shape,X_val.shape,X_test.shape)
        """),
        md("## 4. 기준모델과 공통 평가 함수"),
        code("""
        def metrics(y_true,y_pred):
            return {
                'MAE':mean_absolute_error(y_true,y_pred),
                'RMSE':mean_squared_error(y_true,y_pred)**0.5,
                'R2':r2_score(y_true,y_pred),
            }

        persistence_test=y[idx_test-1]
        baseline_metrics=metrics(y[idx_test],persistence_test)
        print('Persistence baseline:',baseline_metrics)
        """),
        md("## 5. SimpleRNN"),
        code("""
        class SequenceRegressor(nn.Module):
            def __init__(self,kind,hidden=24):
                super().__init__()
                recurrent=nn.RNN if kind=='SimpleRNN' else nn.LSTM
                self.rnn=recurrent(input_size=1,hidden_size=hidden,batch_first=True)
                self.fc=nn.Linear(hidden,1)
            def forward(self,x):
                out,_=self.rnn(x)
                return self.fc(out[:,-1,:])

        def fit_model(kind,epochs=5):
            random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
            model=SequenceRegressor(kind)
            optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
            loss_fn=nn.MSELoss()
            loader=DataLoader(TensorDataset(X_train,y_train),batch_size=512,shuffle=False)
            best_state=None; best_val=np.inf; best_epoch=0; history=[]
            for epoch in range(1,epochs+1):
                model.train(); train_losses=[]
                for xb,yb in loader:
                    optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); train_losses.append(loss.item())
                model.eval()
                with torch.no_grad(): val_loss=loss_fn(model(X_val),y_val).item()
                history.append((epoch,float(np.mean(train_losses)),float(val_loss)))
                if val_loss<best_val:
                    best_val=val_loss; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state); model.eval()
            return model,pd.DataFrame(history,columns=['epoch','train_loss','val_loss']),best_epoch,best_val

        simple_rnn,rnn_history,rnn_best_epoch,rnn_best_val=fit_model('SimpleRNN')
        display(rnn_history)
        print('SimpleRNN best epoch:',rnn_best_epoch,'best val loss:',rnn_best_val)
        """),
        md("## 6. LSTM"),
        code("""
        lstm,lstm_history,lstm_best_epoch,lstm_best_val=fit_model('LSTM')
        display(lstm_history)
        print('LSTM best epoch:',lstm_best_epoch,'best val loss:',lstm_best_val)
        """),
        md("## 7. Validation 비교 후 최종 Test 평가"),
        code("""
        def predict_original(model,X):
            model.eval()
            with torch.no_grad(): pred_scaled=model(X).numpy().ravel()
            return scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()

        rnn_val_pred=predict_original(simple_rnn,X_val)
        lstm_val_pred=predict_original(lstm,X_val)
        validation_table=pd.DataFrame([
            {'Model':'SimpleRNN',**metrics(y[idx_val],rnn_val_pred)},
            {'Model':'LSTM',**metrics(y[idx_val],lstm_val_pred)},
        ]).set_index('Model')
        display(validation_table.round(4))
        selected_model=validation_table['RMSE'].idxmin()
        print('Validation RMSE 기준 선택:',selected_model)

        # 여기서 처음으로 최종 Test를 평가합니다. 두 모델은 과제에서 미리 지정된 비교 대상입니다.
        rnn_test_pred=predict_original(simple_rnn,X_test)
        lstm_test_pred=predict_original(lstm,X_test)
        test_table=pd.DataFrame([
            {'Model':'LastValueBaseline',**baseline_metrics},
            {'Model':'SimpleRNN',**metrics(y[idx_test],rnn_test_pred)},
            {'Model':'LSTM',**metrics(y[idx_test],lstm_test_pred)},
        ]).set_index('Model')
        display(test_table.round(4))
        """),
        md("## 8. 실제값과 예측값 비교"),
        code("""
        view=300
        ts=df['timestamp_corrected'].iloc[idx_test[-view:]]
        plt.figure(figsize=(14,5))
        plt.plot(ts,y[idx_test[-view:]],label='Actual',linewidth=1.5)
        plt.plot(ts,rnn_test_pred[-view:],label='SimpleRNN',alpha=.85)
        plt.plot(ts,lstm_test_pred[-view:],label='LSTM',alpha=.85)
        plt.title('Steel industry Usage_kWh: actual vs recurrent forecasts')
        plt.ylabel('Usage_kWh'); plt.xlabel('time'); plt.legend(); plt.tight_layout(); plt.show()
        """),
        md("""
        ## 9. 결론

        - Validation RMSE로 체크포인트와 모델 우열을 비교하고, Test는 마지막에 평가했습니다.
        - `SimpleRNN`과 `LSTM` 모두 직전값 기준모델과 함께 비교하여 복잡한 모델이 실제로 의미가 있는지 확인했습니다.
        - 이 실습에서는 15분 뒤 1-step 예측으로 가정했습니다. 과제/현업에서 다른 horizon이 필요하면 sequence/target 구성부터 다시 정의해야 합니다.
        - 실제 운영 판단에는 단일 holdout 외에도 rolling-origin backtest, 여러 seed, 업무 비용, 미래 시점에 사용할 수 있는 외생변수 정의가 더 필요합니다.
        """),
        code("""
        print('최종 요약')
        print('- selected by validation RMSE:',selected_model)
        print(test_table.round(4))
        print('- timestamp midnight repairs:',int(midnight_fix.sum()))
        print('- final status: REVIEW (forecast horizon was inferred as one-step ahead for this exercise)')
        """),
    ]
    notebook=nbformat.v4.new_notebook(metadata=original.metadata.copy()); notebook.cells=cells
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    output_path.parent.mkdir(parents=True,exist_ok=True); nbformat.write(notebook,output_path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); ap.add_argument('--target',default='Usage_kWh'); ap.add_argument('--timestamp',default='date'); ap.add_argument('--sequence-length',type=int,default=32)
    a=ap.parse_args(); build(Path(a.input),Path(a.data),Path(a.output),a.target,a.timestamp,a.sequence_length)

if __name__=='__main__': main()
