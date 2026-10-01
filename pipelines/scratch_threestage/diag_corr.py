"""① 纯诊断 — 因子/信号相关矩阵 (只诊断, 不拼装任何组合)。

冻结因子表(long 18 + short 18 去重后的并集) + 方向分数 z = pL - pS,
在2024H1(验证段)上计算 Pearson 与 Spearman 相关。
输出: 相关矩阵PNG + |rho|>0.7 高冗余对清单 + 因子与z的相关排序。
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import config as C


def main():
    F = pd.read_pickle(C.CACHE / 'feats.pkl')
    sv = pd.read_pickle(C.CACHE / 'scores_val.pkl')
    factors = json.load(open(C.RESULTS / 'factors.json'))

    cols = sorted({d['factor'] for side in factors for d in factors[side]})
    va_idx = sv.index
    X = F.loc[va_idx, cols].copy()
    X['z_dir'] = sv['pL'] - sv['pS']
    ok = X.notna().all(axis=1)
    X = X[ok]
    print('诊断样本(2024H1, 因子与z全非NaN):', len(X))

    pear = X.corr(method='pearson')
    spear = X.corr(method='spearman')

    # 高冗余对
    pairs = []
    c = pear.values
    for i in range(len(cols) + 1):
        for j in range(i + 1, len(cols) + 1):
            if abs(c[i, j]) > 0.7:
                pairs.append(dict(a=X.columns[i], b=X.columns[j],
                                  pear=round(float(c[i, j]), 3),
                                  spear=round(float(spear.values[i, j]), 3)))
    pairs.sort(key=lambda d: -abs(d['pear']))
    print(f'\n|r|>0.7 高冗余对: {len(pairs)}')
    for p in pairs:
        print(f"  {p['a']:<20s} {p['b']:<20s} pear={p['pear']:+.3f} spear={p['spear']:+.3f}")

    # 因子与方向分数z的相关(哪些因子在驱动信号)
    zc = spear['z_dir'].drop('z_dir').sort_values(key=np.abs, ascending=False)
    print('\n因子与 z_dir 的Spearman相关(驱动信号的因子):')
    print(zc.round(3).to_string())

    # 矩阵图
    n = pear.shape[0]
    fig, ax = plt.subplots(figsize=(1.0 + 0.32 * n, 1.0 + 0.30 * n),
                           constrained_layout=True)
    im = ax.imshow(pear.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(pear.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(pear.columns, fontsize=7)
    for i in range(n):
        for j in range(n):
            if abs(pear.values[i, j]) > 0.7 and i != j:
                ax.text(j, i, '·', ha='center', va='center',
                        color='black', fontsize=9)
    ax.set_title('冻结因子与方向分数相关矩阵 — 2024H1内部验证段(纯诊断)', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(C.RESULTS / 'diag_corr_matrix.png', dpi=150)
    pear.to_csv(C.RESULTS / 'diag_corr_pearson.csv')
    with open(C.RESULTS / 'diag_corr.json', 'w') as f:
        json.dump({'n_samples': int(len(X)), 'high_corr_pairs': pairs,
                   'corr_with_z': {k: round(float(v), 4) for k, v in zc.items()}},
                  f, indent=2, ensure_ascii=False)
    print('\ndiag_corr_matrix.png / diag_corr.json 已落盘')


if __name__ == '__main__':
    main()
