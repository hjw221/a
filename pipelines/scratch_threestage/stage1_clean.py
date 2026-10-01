"""阶段1a — ML数据清洗。

IsolationForest 异常评分(仅用2023拟合, 应用到整个工作窗 2022-12~2024-07-15)
+ 结构性质量标志(零成交/会话断裂)。输出逐bar质量表与训练清洗掩码。

异常特征向量(全部因果, 单bar粒度):
  |ret1|$, range$, body_ratio, log1p(volume), |ret5|$, 距上根bar的间隔(分钟)
标准化: 2023 的 median/IQR (稳健, 仅训练期统计)。
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import config as C
import dataio

WORK_START = C.WARMUP_START
WORK_END = C.OUTCOME_PAD_END


def build_anomaly_inputs(w: pd.DataFrame) -> pd.DataFrame:
    ret1 = w['close'] - w['open']
    prev_close = w['close'].shift(1)
    ret5 = w['close'] - prev_close.shift(4)
    rng = w['high'] - w['low']
    body = (w['close'] - w['open']).abs()
    gap_min = w.index.to_series().diff().dt.total_seconds().div(60).fillna(1.0)
    x = pd.DataFrame({
        'abs_ret1': ret1.abs(),
        'range': rng,
        'body_ratio': (body / rng.replace(0, np.nan)).fillna(0.0),
        'log_vol': np.log1p(w['volume']),
        'abs_ret5': ret5.abs(),
        'gap_min': gap_min.clip(0, 120),
    })
    return x


def main():
    df = dataio.load_raw()
    w = df.loc[WORK_START:WORK_END]
    print('工作窗:', len(w), 'bars', w.index[0], '->', w.index[-1])

    x = build_anomaly_inputs(w)
    tr_mask_2023 = (x.index >= C.DATA_START) & (x.index <= C.TRAIN_END)
    x23 = x[tr_mask_2023]

    # 稳健标准化(统计仅来自2023)
    med = x23.median()
    iqr = (x23.quantile(0.75) - x23.quantile(0.25)).replace(0, 1.0)
    z = ((x - med) / iqr).clip(-20, 20).fillna(0.0)  # 首行shift的NaN按中位数填充

    # IsolationForest 仅在2023(子采样)拟合
    rng = np.random.RandomState(C.SEED)
    sub = z[tr_mask_2023]
    if len(sub) > 150_000:
        idx = rng.choice(len(sub), 150_000, replace=False)
        sub = sub.iloc[idx]
    iso = IsolationForest(n_estimators=200, contamination=0.01,
                          random_state=C.SEED, n_jobs=C.NJOBS)
    iso.fit(sub)
    score = iso.decision_function(z.values)          # 越小越异常
    anomaly = (iso.predict(z.values) == -1)

    q = pd.DataFrame({'iso_score': score, 'anomaly': anomaly}, index=w.index)
    q['vol0'] = w['volume'] <= 0
    gap_min = w.index.to_series().diff().dt.total_seconds().div(60)
    q['gap'] = gap_min.fillna(1.0) > 5.0              # 会话/周末断裂后的第一根bar
    q['clean'] = (~q['anomaly']) & (~q['vol0'])

    C.CACHE.mkdir(exist_ok=True)
    q.to_pickle(C.CACHE / 'quality.pkl')

    # ---- 摘要 ----
    res = {'work_window': [str(w.index[0]), str(w.index[-1])],
           'n_bars': int(len(w)),
           'n_anomaly': int(q['anomaly'].sum()),
           'n_vol0': int(q['vol0'].sum()),
           'n_gap_bar': int(q['gap'].sum()),
           'clean_ratio_work': float(q['clean'].mean())}
    for name, s, e in [('core_train_2023', C.DATA_START, C.TRAIN_END),
                       ('val_2024H1', C.VAL_START, C.VAL_END)]:
        qq = q.loc[s:e]
        res[f'clean_ratio_{name}'] = float(qq['clean'].mean())
        res[f'n_anomaly_{name}'] = int(qq['anomaly'].sum())
    # 异常的时段分布(哪些数据脏)
    q23 = q.loc[C.DATA_START:C.TRAIN_END]
    byh = q23.groupby(q23.index.hour)['anomaly'].mean()
    res['anomaly_rate_by_hour_2023'] = {int(h): round(float(v), 4) for h, v in byh.items()}
    # 最异常的20个时间点
    worst = q23.nsmallest(20, 'iso_score')
    res['top20_anomalies'] = [str(t) for t in worst.index]

    C.RESULTS.mkdir(exist_ok=True)
    with open(C.RESULTS / 'stage1_clean.json', 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main()
