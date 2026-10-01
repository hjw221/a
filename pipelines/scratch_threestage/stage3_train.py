"""阶段3(v2) — 单模型训练(方法论最后一步)。

修正要点(相对v1):
  - 早停改用2023Q4: 2024H1 不参与任何训练/选择, 纯粹留给消融
  - 拟合段=2023-01~2023-09, 指数时间衰减权重 tau=120天(应对2023->2024波动regime漂移)
  - 保存拟合段分数分布 -> 冻结分位数阈值表(供δ收紧消融, 尺度无关)
结构注记: 该几何下 yL 与 yS 近互补(超时≈0), 方向预测本质一维, 双模型近似镜像(保留以容差校准差)。
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config as C
import dataio

FIT_END = '2023-09-30 23:59:59'
ES_END = C.TRAIN_END            # 2023Q4 早停段
TAU_DAYS = 120                  # 时间衰减
PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31,
              min_child_samples=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              random_state=C.SEED, n_jobs=C.NJOBS, verbose=-1)


def main():
    F = pd.read_pickle(C.CACHE / 'feats.pkl')
    L = pd.read_pickle(C.CACHE / 'labels.pkl')
    Q = pd.read_pickle(C.CACHE / 'quality.pkl')
    geom = json.load(open(C.CACHE / 'geometry.json'))
    factors = json.load(open(C.RESULTS / 'factors.json'))
    H = int(geom['H'])

    idx = F.index
    fit_time = (idx >= np.datetime64(C.DATA_START)) & (idx <= np.datetime64(FIT_END))
    es_time = (idx > np.datetime64(FIT_END)) & (idx <= np.datetime64(ES_END))
    va_time = (idx >= np.datetime64(C.VAL_START)) & (idx <= np.datetime64(C.VAL_END))
    feasible = (L['codeL'].values != 2)
    common_ok = feasible & Q['clean'].values & L['universe'].values

    t_ref = pd.Timestamp(FIT_END)
    age_days = np.asarray((t_ref - idx).total_seconds() / 86400.0)
    w = np.exp(-np.clip(age_days, 0, None) / TAU_DAYS)   # 拟合段内近期权重高

    scores, summary = {}, {'geometry': f"kE={geom['kE']} PLR={geom['plr']} H={H}",
                           'fit_window': [C.DATA_START, FIT_END],
                           'es_window': [FIT_END, ES_END],
                           'tau_days': TAU_DAYS}
    mon = pd.Series(idx.strftime('%Y-%m'))
    for side in ('long', 'short'):
        cols = [d['factor'] for d in factors[side]]
        X = F[cols]
        y = (L[f'code{side[0].upper()}'].values == 1)
        m_fit = fit_time & common_ok & X.notna().all(axis=1).values
        # fit尾purge: 标签不读入早停段
        fit_end_pos = int(np.searchsorted(idx.values, np.datetime64(FIT_END)))
        m_fit &= (np.arange(len(idx)) < fit_end_pos - H)
        m_es = es_time & common_ok & X.notna().all(axis=1).values
        m_va = va_time & common_ok & X.notna().all(axis=1).values
        print(f'[{side}] 训练 {m_fit.sum():,} (有效权重 {w[m_fit].sum():,.0f}) | '
              f'早停 {m_es.sum():,} | 验证 {m_va.sum():,} | 基率 '
              f'{y[m_fit].mean():.3f}/{y[m_es].mean():.3f}/{y[m_va].mean():.3f}')

        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[m_fit], y[m_fit], sample_weight=w[m_fit],
              eval_set=[(X[m_es], y[m_es])], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        p_fit = m.predict_proba(X[m_fit])[:, 1]
        p_es = m.predict_proba(X[m_es])[:, 1]
        p_va = m.predict_proba(X[m_va])[:, 1]
        print(f'[{side}] best_iter={m.best_iteration_} '
              f'AUC fit={roc_auc_score(y[m_fit], p_fit):.4f} '
              f'es(2023Q4)={roc_auc_score(y[m_es], p_es):.4f} '
              f'val(2024H1)={roc_auc_score(y[m_va], p_va):.4f}')

        # 2024H1 逐月AUC(迁移稳定性)
        mon_val = mon[m_va].to_numpy()
        y_va, m_va_mask = y[m_va], m_va
        monthly = {}
        for mo in sorted(set(mon_val)):
            mk = (mon_val == mo)
            if len(np.unique(y_va[mk])) > 1:
                monthly[mo] = round(float(roc_auc_score(y_va[mk], p_va[mk])), 4)
        print(f'[{side}] 2024H1逐月AUC: {monthly}')

        # 冻结分位数阈值(拟合段分数分布) — δ收紧消融的刻度
        qs = {d: float(np.quantile(p_fit, 1 - d))
              for d in (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)}
        print(f'[{side}] 冻结阈值(2023拟合段分位): '
              + ', '.join(f'δ={k:.2f}->{v:.4f}' for k, v in qs.items()))

        # 验证段校准(十分位)
        q = pd.DataFrame({'p': p_va, 'y': y[m_va]})
        q['b'] = pd.qcut(q['p'], 10, duplicates='drop')
        cal = q.groupby('b', observed=True).agg(n=('y', 'size'), p_mean=('p', 'mean'),
                                                hit=('y', 'mean'))
        print(f'[{side}] 2024H1十分位校准: hit base={y[m_va].mean():.3f}')
        print(cal.to_string())

        scores[side] = dict(fit=(idx[m_fit], p_fit), val=(idx[m_va], p_va))
        m.booster_.save_model(str(C.CACHE / f'model_{side}.txt'))
        summary[side] = dict(
            n_fit=int(m_fit.sum()), n_es=int(m_es.sum()), n_val=int(m_va.sum()),
            best_iter=int(m.best_iteration_),
            auc_fit=round(float(roc_auc_score(y[m_fit], p_fit)), 4),
            auc_es=round(float(roc_auc_score(y[m_es], p_es)), 4),
            auc_val=round(float(roc_auc_score(y[m_va], p_va)), 4),
            monthly_auc_val=monthly, thresholds=qs,
            base_val=round(float(y[m_va].mean()), 4),
            calibration=[dict(bin=str(b), n=int(r['n']), p_mean=round(float(r['p_mean']), 4),
                             hit=round(float(r['hit']), 4)) for b, r in cal.iterrows()])

    sv = pd.DataFrame({'pL': pd.Series(scores['long']['val'][1], index=scores['long']['val'][0]),
                       'pS': pd.Series(scores['short']['val'][1], index=scores['short']['val'][0])})
    sv.to_pickle(C.CACHE / 'scores_val.pkl')
    pd.DataFrame({'pL': pd.Series(scores['long']['fit'][1], index=scores['long']['fit'][0]),
                  'pS': pd.Series(scores['short']['fit'][1], index=scores['short']['fit'][0])}) \
        .to_pickle(C.CACHE / 'scores_fit.pkl')
    with open(C.RESULTS / 'stage3_train.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('模型/scores已落盘')


if __name__ == '__main__':
    main()
