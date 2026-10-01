#!/usr/bin/env python3
"""从 v3bal_ens 真实逐笔日志计算 MT5 可测子区间的 Python 参考指标。
用途: 用户 MT5 回测窗口受经纪商 M5 历史深度限制(实测 Exness M5 从 2025-01 起),
      需要对应子区间的真实基准做对照。只读真实数据, 不做任何模拟。"""
import numpy as np
import pandas as pd

REF = ('/home/z/my-project/download/xauusd_ml_v2/mt5_package/'
       'reference/trades_v3bal_ens_python.csv')
OUT = ('/home/z/my-project/download/xauusd_ml_v2/mt5_package/'
       'reference/mt5_subset_reference.csv')

df = pd.read_csv(REF)
df['signal_time'] = pd.to_datetime(df['signal_time'])
assert len(df) == 1611, f'expect 1611 trades, got {len(df)}'
assert abs(df.pnl.sum() - 1113.5) < 0.5, f'total pnl {df.pnl.sum():.1f} != 1113.5'


def stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    pnl = sub.pnl.values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr = len(wins) / n if n else np.nan
    plr = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else np.nan
    # 最长连亏
    mcl = cur = 0
    for p in pnl:
        cur = cur + 1 if p <= 0 else 0
        mcl = max(mcl, cur)
    # 逐笔口径 maxDD
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    lo = sub[sub.dir == 'long']
    sh = sub[sub.dir == 'short']
    return dict(n=n, wr=100 * wr, plr=plr, pnl=pnl.sum(), avg=pnl.mean(),
                mcl=mcl, mdd=mdd,
                n_long=len(lo), pnl_long=lo.pnl.sum(),
                n_short=len(sh), pnl_short=sh.pnl.sum())


MONTHS = sorted(df.fold.unique())
subsets = [
    ('全24个月 2024-08~2026-07', MONTHS),
    ('19个月 2025-01~2026-07', [m for m in MONTHS if m >= '2025-01']),
    ('7个月 2026-01~2026-07', [m for m in MONTHS if m >= '2026-01']),
    ('2个月 2026-06~2026-07', [m for m in MONTHS if m >= '2026-06']),
]

rows = []
print(f"{'区间':<26}{'笔数':>6}{'胜率%':>8}{'PLR':>7}{'总PnL$':>10}"
      f"{'单笔$':>8}{'连亏':>6}{'maxDD$':>8}{'多/空笔':>10}{'多$/空$':>16}")
for name, months in subsets:
    sub = df[df.fold.isin(months)]
    s = stats(sub)
    rows.append(dict(window=name, **s))
    print(f"{name:<26}{s['n']:>6}{s['wr']:>8.1f}{s['plr']:>7.2f}{s['pnl']:>10.1f}"
          f"{s['avg']:>8.2f}{s['mcl']:>6}{s['mdd']:>8.0f}"
          f"{s['n_long']}/{s['n_short']:>4}"
          f"{s['pnl_long']:>8.1f}/{s['pnl_short']:.1f}")

# 预热缺口: 若测试起点2025-01-01且M5历史恰从2025-01-01起, 前~3001根M5(约12天)无法评估
warm = df[(df.fold == '2025-01') & (df.signal_time < '2025-01-13')]
print(f"\n[预热缺口] 2025-01-01~01-12(测试起点即历史起点时无法评估): "
      f"{len(warm)}笔, PnL ${warm.pnl.sum():.1f}")
# 逐月PnL供对照
mo = df.groupby('fold').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'))
print('\n[逐月] 2025-01之后的月份(用户可完整复现的部分):')
for m, r in mo[mo.index >= '2025-01'].iterrows():
    print(f"  {m}: {int(r.n):>4}笔  ${r.pnl:>8.1f}")

pd.DataFrame(rows).to_csv(OUT, index=False, lineterminator='\r\n')
print(f"\nsaved -> {OUT}")
