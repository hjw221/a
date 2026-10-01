"""② 消融可视化 — δ收紧曲线: 每笔期望/方向alpha/交易数 vs δ (按regime变体)。"""
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

R = pd.read_csv(C.RESULTS / 'ablation_regime_delta.csv')
M = R[R['side'] == 'model'].set_index(['regime', 'delta'])
Rd = R[R['side'] == 'random'].set_index(['regime', 'delta'])

show = ['all', 'wings', 'high', 'low', 'mid']
colors = {'all': '#888888', 'wings': '#B8860B', 'high': '#C0392B',
          'low': '#2471A3', 'mid': '#7D3C98'}
deltas = sorted(R['delta'].unique())

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True,
                         sharex=True)
for rv in show:
    d = deltas
    avg = [M.loc[(rv, x), 'avg'] for x in d]
    alp = [M.loc[(rv, x), 'avg'] - Rd.loc[(rv, x), 'avg'] for x in d]
    n = [M.loc[(rv, x), 'n'] for x in d]
    axes[0].plot(d, avg, 'o-', ms=4, lw=1.6, color=colors[rv], label=rv)
    axes[1].plot(d, alp, 'o-', ms=4, lw=1.6, color=colors[rv], label=rv)
    axes[2].plot(d, n, 'o-', ms=4, lw=1.6, color=colors[rv], label=rv)

axes[0].axhline(0, color='k', lw=0.8, ls='--')
axes[0].set_title('每笔净期望 ($/笔) vs δ收紧', fontsize=11)
axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].set_title('方向alpha (模型-随机, $/笔)', fontsize=11)
axes[2].set_yscale('log')
axes[2].set_title('交易笔数 (log)', fontsize=11)
for ax in axes:
    ax.set_xlabel('δ = 中部不交易带宽度 (0.5=全交易)', fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
axes[0].legend(fontsize=9, frameon=False)
fig.suptitle('② 波动Regime门控 × δ收紧消融 — 2024H1内部验证段 (成本$0.06往返, OOS未触碰)',
             fontsize=12)
fig.savefig(C.RESULTS / 'ablation_delta_curve.png', dpi=150)
print('ablation_delta_curve.png 已落盘')
