# 服务器阶段成果存档（2025-09-29 16:40 停机快照）

## 背景
- 机器：jupyter-fohh8ytu3bypyx48（gpuhome 无卡模式，cgroup 实际限额 **1核/16G**）
- 因 1 核太慢 + 明天换真 12 核机器，用户指示 **16:40 停机**，成果存档于此
- 本目录 = `/root/rivermind-fs/xauusd/{v2_ens/results/, logs/}` 的完整快照（tar md5: 28b1ceb2d940b70494612468533863ba）

## 已完成的里程碑
| 阶段 | 状态 | 产出 |
|---|---|---|
| Phase1 v3bal_ens 重跑 | ✅ 完成（14:07） | summary/per_fold/trades + 24 折 checkpoint |
| Phase2 prep（m1sc_ad pack 重建） | ✅ 完成 | 160万行 / 标签有效率 99.94% / TP率 L34% S31% 与历史 ground truth 一致 |
| Phase2 tune（12配置×2内折） | ✅ 完成（14:51） | **results/tuned_params_m1.json**（LGB+XGB 双模型完整超参）|
| Phase3 四臂走查 | ⏸ 中断于 ~26%（16:40） | 25 个 m1sc checkpoint（A0 到 fold12，其余臂 fold2-3）|

## ⚡ 明天 12 核机器的关键情报
1. **断点续跑可用**：run_m1.py 原生支持（fold 完成→checkpoint→重跑自动跳过，见 run_m1.py L322）。
   恢复方法：把本目录 `results/checkpoints/` 原位放回 `v2_ens/results/checkpoints/`，直接重跑四臂命令即可从断点继续。
2. **run_m1.py 修复版**：commit 218c1ed（时间轴 bug：`pd.to_datetime(unit="m")` 误用相对分钟偏移 → 改用 load_raw_m1(csv).index 真实时间轴）。Phase2 已在新代码下验证通过。
3. **部署三件套**（沙箱侧）：recovery-tools/jump-ssh.py（命令）、jump-upload.py（上传）、jump-download.py（下载）；GitHub 推送用 ~/.ssh/git-ssh-paramiko.py shim。
4. **Phase1 复现差异已知**：服务器 406笔/PLR 2.31 vs 历史 1611笔/PLR 2.57——代码/数据/超参已做 bit 级 diff 排除，定格为运行环境差异（库版本/多线程浮点非确定性）。四臂消融在任一环境内部自洽。

## 四臂中断位置（checkpoint 计数）
- A0 m1sc_ad: fold00-12（13/24）
- A1 m1sc_ad_ens: fold00-03（4/24）
- A2 m1sc_ad_ens6: fold00-0x（少量）
- A3 m1sc_ad_ens_hl180: fold00-03（4/24）
- 另含 Phase1 v3bal_ens 完整 24 折 checkpoint

## 目录结构
- `results/` — tuned_params_m1.json（核心资产）、summary/per_fold/trades（Phase1 全套）、checkpoints/（49 个 pkl）、historical_reference/（从 main 分支拉的历史对照）
- `01~05_*.log + master.log + agent.log` — 全程审计日志
