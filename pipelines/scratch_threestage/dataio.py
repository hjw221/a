"""数据层: 原始M1 CSV -> 重建月度文件(download/xauusd_data) + 统一缓存。

零旧代码复用。列: time, open, high, low, close, volume(=TICKVOL), spread。
"""
import pandas as pd
import numpy as np
import config as C


def load_raw() -> pd.DataFrame:
    """读取原始CSV, 去重排序, 统一索引, 缓存为pickle。"""
    f = C.CACHE / 'm1_all.pkl'
    if f.exists():
        return pd.read_pickle(f)
    df = pd.read_csv(C.SRC_CSV, sep='\t')
    df.columns = [c.strip().strip('<>').lower() for c in df.columns]
    t = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str),
                       format='%Y.%m.%d %H:%M:%S')
    out = df[['open', 'high', 'low', 'close', 'tickvol', 'spread']].copy()
    out.index = t                       # 直接赋索引, 避免dict构造触发标签对齐
    out = out.rename(columns={'tickvol': 'volume'})
    out = out[~out.index.duplicated(keep='last')].sort_index()
    for c in out.columns:
        out[c] = out[c].astype('float64')
    assert not out[['open', 'high', 'low', 'close']].isna().any().any(), 'OHLC存在NaN'
    C.CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(f)
    return out


def write_monthly(df: pd.DataFrame) -> int:
    """重建月度文件 XAUUSD_M1_YYYYMM.csv (列: time,open,high,low,close,volume)。"""
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    sub = df.loc[C.DATA_START:]
    n = 0
    for ym, g in sub.groupby(sub.index.to_period('M')):
        p = C.DATA_DIR / f'XAUUSD_M1_{ym}.csv'
        if not p.exists():
            g.rename_axis('time').reset_index()[
                ['time', 'open', 'high', 'low', 'close', 'volume']] \
                .to_csv(p, index=False)
        n += 1
    return n


def load_monthly() -> pd.DataFrame:
    """从重建的月度文件读取(验证数据层可用, 与load_raw一致性校验)。"""
    files = sorted(C.DATA_DIR.glob('XAUUSD_M1_*.csv'))
    frames = []
    for p in files:
        d = pd.read_csv(p)
        d['time'] = pd.to_datetime(d['time'])
        frames.append(d.set_index('time'))
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep='last')].sort_index()


if __name__ == '__main__':
    df = load_raw()
    print('raw bars:', len(df), '|', df.index[0], '->', df.index[-1])
    n = write_monthly(df)
    print('monthly files:', n)
    # 数据层一致性校验
    m = load_monthly()
    assert len(m) == len(df.loc[C.DATA_START:]), 'monthly row count mismatch'
    ok = np.allclose(m['close'].values, df.loc[C.DATA_START:, 'close'].values)
    print('monthly vs raw close allclose:', ok)
    # 切分统计
    for name, s, e in [
        ('warmup        ', C.WARMUP_START, '2022-12-31 23:59:59'),
        ('core_train 2023', C.DATA_START, C.TRAIN_END),
        ('val        2024H1', C.VAL_START, C.VAL_END),
        ('oos(不动)  2024-07+', C.OOS_START, df.index[-1]),
    ]:
        g = df.loc[s:e]
        print(f'{name}: {len(g):,} bars  {g.index[0]} -> {g.index[-1]}')
    # 分钟连续性
    dts = df.loc[C.DATA_START:].index.to_series().diff().dt.total_seconds()
    print('bars gap>60min 次数(周末/假日):', int((dts > 60).sum()),
          '| 最大gap(分钟):', dts.max() / 60)
    print('2023 spread列非零比例:', float((df.loc[C.DATA_START:C.TRAIN_END, 'spread'] > 0).mean()))
