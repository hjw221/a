# XAUUSD M1 量化模型 — 服务器工作区

> 持久化桶 `/root/rivermind-fs`，GPU: RTX 3090（注: LightGBM/XGBoost 树模型在此数据规模下为 CPU 负载，GPU 不参与，保证与本地口径一致）
> 建立日期: 2026-09-29

## 目录结构

```
xauusd/
├── README.md                     # 本文件
├── setup_env.sh                  # 依赖安装 (lightgbm==4.5.0, xgboost==2.1.3, ...)
├── data/
│   └── XAUUSDc_M1_*.csv          # 原始 M1 数据 (2022-01~2026-07, 160万根)
├── v2_ens/                       # v2 管线 (冠军 v3bal_ens + m1sc_ad)
│   ├── run_all.py                #   编排器: prep/tune/wf (--features v3bal, v3bal_ens, ...)
│   ├── run_m1.py                 #   m1sc_ad 重建+续训驱动 (4个预注册臂)
│   ├── models.py                 #   已patch: ML_THREADS 环境变量控制线程 (默认2)
│   ├── config.py                 #   data_path 已指向 ../data/*.csv
│   └── results/                  #   tuned_params*.json (冻结超参) + 走查输出
├── logs/                         # 全部运行日志 (编号前缀=执行顺序)
└── run_phase{1,2,3}_*.sh        # 三阶段启动脚本
```

## 执行顺序

```bash
cd /root/rivermind-fs/xauusd
bash setup_env.sh                     # ① 依赖
bash run_phase1_v3bal.sh              # ② 重跑冠军 v3bal_ens (~30分钟)
bash run_phase2_m1sc_prep.sh          # ③ m1sc_ad: 特征标签 + 重新调参 (~1.5小时)
bash run_phase3_arms.sh               # ④ 四臂并行走查 (~4-8小时, 断点续跑)
```

## 任务说明

1. **Phase 1 — v3bal_ens 重跑**: 与本地历史（1611笔/31.6%/PLR 2.57/+$1,113.5/Sharpe 1.74）严格同口径复现。
   超参来自 `results/tuned_params.json`["v3"]（冻结），pack 在服务器重建。
2. **Phase 2 — m1sc_ad 重建**: 原始 m1sc_ad 的超参随被删的驱动脚本一起丢失（pack 中的特征/标签/几何已
   逆向验证到 1e-15 精度，完全恢复）。本阶段按原协议（12配置×2内折×子采样4，首训练窗内）重新调参。
3. **Phase 3 — 四臂续训**（预注册，严格消融，全部真实 OOS 2024-08~2026-07）:
   | 臂 | 说明 | 针对的弱点 |
   |---|---|---|
   | A0 `m1sc_ad` | 基线重建 | — |
   | A1 `m1sc_ad_ens` | 3种子×lgb/xgb AUC加权集成 | PLR 2.05 / 方差 |
   | A2 `m1sc_ad_ens6` | 6种子深集成 | 同上 |
   | A3 `m1sc_ad_ens_hl180` | 半衰期270→180d | 利润集中 2026-02/03 |

## 结果对照基准 (历史真实数字)

| 模型 | 笔数 | 胜率 | PLR | PnL | Sharpe |
|---|---:|---:|---:|---:|---:|
| v3bal_ens (本地历史) | 1,611 | 31.6% | 2.57 | +$1,113.5 | 1.74 |
| m1sc_ad (原始, 超参已丢) | 6,496 | 35.2% | 2.05 | +$732.7 | 2.29 |
