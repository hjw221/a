"""阶段2 — ML因子挖掘(只用2023, 内部再分H1拟合/H2验证, 尾部按H purge)。

每个候选因子 x 每个方向(long/short):
  1) 单因子LGBM AUC (拟合2023H1, 验证2023H2) — 与最终目标(TP-first)同口径
  2) 月度RankIC: 因子 vs 规格化路径质量 gross/ATR60 的Spearman相关(TP=+3, SL=-1.5常数化,
     剔除障碍尺寸机械通道, 只剩方向+路径信息), 按月计算 -> 均值/t值(稳定性)
  3) 综合分 = 0.5*AUC排名 + 0.25*|IC|排名 + 0.25*IC-t排名
家族去重(同前缀最多3个) -> 每方向冻结 top18, 写入 results/factors.json
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import config as C
import dataio


def main():
    F = pd.read_pickle(C.CACHE / 'feats.pkl')
    L = pd.read_pickle(C.CACHE / 'labels.pkl')
    Q = pd.read_pickle(C.CACHE / 'quality.pkl')
    geom = json.load(open(C.CACHE / 'geometry.json'))
    H = int(geom['H'])

    idx = F.index
    m23 = (idx >= pd.Timestamp(C.DATA_START)) & (idx <= pd.Timestamp(C.TRAIN_END))
    pos = np.arange(len(idx))
    # 尾部purge: 标签窗口不得越过2023
    train_end_pos = int(np.searchsorted(idx.values, np.datetime64(C.TRAIN_END)))
    purge = pos < train_end_pos - H
    base = m23 & purge & (L['codeL'].values != 2) & Q['clean'].values & L['universe'].values
    base = base & (~F.iloc[:, 0].isna().values)
    base = base & (L['atr60'].values > 0)
    sub = base.copy()
    sub[::2] = False          # 隔一根采样, 降低标签重叠带来的伪稳定
    print('筛选样本: 全量 %d, 采样后 %d' % (base.sum(), sub.sum()))

    fit_m = sub & (idx < np.datetime64('2023-07-01'))
    val_m = sub & (idx >= np.datetime64('2023-07-01'))
    months = pd.Series(idx.strftime('%Y-%m'))
    yL = (L['codeL'].values == 1)
    yS = (L['codeS'].values == 1)
    # 规格化路径质量: 剔除障碍随ATR缩放的机械尺寸通道, 保留方向+路径信息
    qL = L['pnlL_gross'].values / L['atr60'].values
    qS = L['pnlS_gross'].values / L['atr60'].values

    out = {}
    for side, y, pnl in (('long', yL, qL), ('short', yS, qS)):
        rows = []
        for col in F.columns:
            x = F[col].values
            ok = np.isfinite(x) & sub
            xs, ys = x[ok], y[ok]
            if np.std(xs) == 0 or len(np.unique(ys)) < 2:
                rows.append(dict(factor=col, auc=np.nan, ic_mean=np.nan, ic_t=np.nan))
                continue
            okf = np.isfinite(x) & fit_m
            okv = np.isfinite(x) & val_m
            if len(np.unique(y[okf])) < 2 or len(np.unique(y[okv])) < 2:
                rows.append(dict(factor=col, auc=np.nan, ic_mean=np.nan, ic_t=np.nan))
                continue
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=15,
                                    min_child_samples=500, random_state=C.SEED,
                                    n_jobs=C.NJOBS, verbose=-1)
            m.fit(x[okf].reshape(-1, 1), y[okf],
                  eval_set=[(x[okv].reshape(-1, 1), y[okv])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            auc = float(roc_auc_score(y[okv],
                                      m.predict_proba(x[okv].reshape(-1, 1))[:, 1]))
            ics = []
            for mo in sorted(set(months[ok])):
                mk = ok & (months == mo).values
                if mk.sum() < 1000 or len(np.unique(y[mk])) < 2:
                    continue
                ic = spearmanr(x[mk], pnl[mk]).statistic
                if np.isfinite(ic):
                    ics.append(ic)
            ics = np.array(ics)
            icm = float(ics.mean()) if len(ics) else np.nan
            ict = float(ics.mean() / (ics.std() / np.sqrt(len(ics)))) if len(ics) > 1 else np.nan
            rows.append(dict(factor=col, auc=round(auc, 4),
                             ic_mean=round(icm, 4) if np.isfinite(icm) else None,
                             ic_t=round(ict, 2) if np.isfinite(ict) else None,
                             n_months=int(len(ics))))
        t = pd.DataFrame(rows).dropna(subset=['auc'])
        t['r_auc'] = t['auc'].rank(ascending=False)
        t['r_ic'] = t['ic_mean'].abs().rank(ascending=False)
        t['r_ict'] = t['ic_t'].abs().rank(ascending=False)
        t['score'] = 0.5 * (len(t) - t['r_auc']) / len(t) \
            + 0.25 * (len(t) - t['r_ic']) / len(t) \
            + 0.25 * (len(t) - t['r_ict']) / len(t)
        t['family'] = t['factor'].str.split('_').str[0]
        t = t.sort_values('score', ascending=False)
        # 家族去重: 每家族最多3个
        keep_rows, cnt = [], {}
        for _, r in t.iterrows():
            f = r['family']
            cnt.setdefault(f, 0)
            if cnt[f] < 3:
                keep_rows.append(r)
                cnt[f] += 1
            if len(keep_rows) >= 18:
                break
        sel = pd.DataFrame(keep_rows)
        out[side] = json.loads(sel.to_json(orient='records'))
        print(f'--- {side}: 单因子AUC前10 ---')
        print(t.sort_values('auc', ascending=False).head(10)
              [['factor', 'auc', 'ic_mean', 'ic_t']].to_string(index=False))
        print(f'--- {side}: 冻结因子(综合分, 家族去重top18) ---')
        print(sel[['factor', 'auc', 'ic_mean', 'ic_t', 'score']].to_string(index=False))
        t.to_csv(C.RESULTS / f'factor_screen_{side}.csv', index=False)

    with open(C.RESULTS / 'factors.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('factors.json 已冻结')


if __name__ == '__main__':
    main()
