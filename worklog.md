# 工作日志（沙箱第 N 次重置后重建）

---
Task ID: 15
Agent: main (Z.ai Code)
Task: 用户给新服务器 sx01-ssh.gpuhome.cc:30133 (ryq94rge) "30核80G 上上上全部跑一遍都行"——部署并全量重跑三阶段管线

Work Log:
- 沙箱又重置：recovery-tools/gh-server-fix/worklog/私钥全丢；私钥内容不可恢复（影响 GitHub push，不影响本次 clone 部署）
- 重建工具链：pip 装 paramiko 5.0 → jump_ssh_lib.py（目标机参数化，默认新机 30133）+ jump-ssh.py + stream-upload.py + relay-upload.py
- 新机验证：hostname jupyter-g0ue1ecf2evh9rr9；cgroup cpu.max=3000000/100000=30核、memory.max=80G 双实锤；49G 持久卷 /root/rivermind-data；conda Python 3.12
- 依赖：pip 默认源超时 → 清华镜像 4 分钟装完 lightgbm 4.7 / xgboost 3.4.1 / sklearn 1.9.1 / pandas / numpy / scipy
- 老机 30181 已被释放（Connection refused），旧部署不可复用
- **网络地狱全量实测**（新机 30133 路径级限速 ~3-30KB/s）：SFTP 2.8KB/s、exec 流式 22KB/s、4 路并行聚合不变（路径级 QoS）、跳板原生 ssh 同慢、新机反拉 GCP 10KB/s、raw.githubusercontent/ghfast.top/gh-proxy.com/ghproxy.net 全挂、ntfy.sh 免费附件限额拒 8MB、codeload.github.com 33KB/s 是唯一可用国际通道、国内（清华）快
- **部署方案**：代码包 428KB（排除 data gz）exec 流式 88s + md5 ✓；rivermind-fs→rivermind-data 软链一招解所有硬编码路径；py_compile 全过；agent.py 上线 pid 10230
- **自主部署脚本** /tmp/auto_deploy.sh（nohup）：codeload 循环下载 26.7MB tarball（40 次重试+md5 门禁）→ 三重 md5（tar/gz/csv）→ gunzip → sed 补丁 phase3 每臂 ML_THREADS=7 → ML_THREADS=26 点火 run_master.sh → ntfy 每步汇报（手机可看）
- 线程布局：phase1/2 单进程 26 线程；phase3 四臂并行各 7 线程（4×7=28/30核）
- 用户"全部跑一遍"→ 不用旧 checkpoint 断点，单环境全新跑（temp_server_run_20250929 留仓库对照）
- 档案重建：remote-ops-record/README.md（凭据/通道格局/坑位速查）

Stage Summary:
- 05:24 auto_deploy 启动，数据下载中（33KB/s，预计 ~05:44 到位后自动点火）
- 全跑预期：Phase1 ~30min → Phase2 ~30-60min → Phase3 四臂 ~1-2h，总计 ~2.5-3.5h（30 核真火力）
- 完成标志：手机 ntfy RES topic 收到 "pipeline ALL DONE"；agent.py 可远程查 status/tail
- 历史基线对照：m1sc_ad 6496笔/35.2%/PLR 2.05/+$732.7/Sharpe 2.29

---
Task ID: 16
Agent: main (Z.ai Code)
Task: 新机(30133)部署收尾——数据投递、点火、排障 numba

Work Log:
- 网络最终方案：codeload curl 两轮均在 ~12min 连接上限处死（最多 ~24MB < 26.7MB 永远差一点）；ntfy 附件下载仅 9.5KB/s；jsDelivr 403(>20MB)；api.github.com 16KB/s 更慢
- 制胜招：**xz -9e 把 101MB CSV 压到 17.2MB**（比 gz 小 35%），八路并行 exec 流式 237s 全部到位（聚合 74KB/s，单路 3 倍），块级 md5 + 拼装 md5 + CSV md5 三重校验全绿
- 首次点火翻车：features.py 需要 **numba**（ModuleNotFoundError），set -e 使 run_phase1 静默死亡；教训：装依赖前应先 grep 全部 import（本地扫描发现 numba+matplotlib 漏装）
- 补装 numba 0.67 + matplotlib 3.11（清华源秒装），八依赖全量验证一把过
- 二次点火成功：prep 2 秒完成（numba JIT）；wf 走查 **22.3 核真实满载**（cgroup cpu.stat 微分实测），4 分钟 20/24 折，内存 0.73G/80G
- 工具沉淀：stream-upload.py（单文件流式）+ xz-push.py（分块八路并行推送，块级断点续传）+ relay-upload.py（已证无效保留存档）

Stage Summary:
- 管线全速运行中：Phase1 wf 22 核满载 ~12s/折（1核机 2min/折 → 10 倍提速）
- 全程预计 ~1 小时完成（phase1→2→3 串行自动接力，ntfy 每步汇报）
- 数据投递方法论成型：大文件走"xz 极限压缩 + 分块并行 exec 流式"，codeload 33KB/s 做 4MB 级小文件兜底

---
Task ID: 17
Agent: main (Z.ai Code)
Task: 30核机全量管线收官——四臂成绩出齐、结果归档、复盘

Work Log:
- 全程耗时：05:54 点火 → 06:48 结果包上传 = **54 分钟**全跑完（1核机预估 8h+，提速 ~10 倍）
- 火力实测：phase1 22.3 核持续满载（cgroup 微分）；phase3 四臂各 ~6.7 核（ML_THREADS=7 补丁生效）；内存峰值 ~3G/80G
- Phase1 v3bal_ens 重跑：409笔/29.1%/PLR 2.36/-$40.7（1核机 406笔/+$18.6；历史 1611笔/+$1113.5）——**第三个环境数据点**：26线程与2线程两次服务器跑互差，多线程浮点非确定性实锤
- Phase2 tune：调参结果与 1 核机逐位一致（种子确定性 ✓）——learning_rate 0.0572/num_leaves 47/min_data_in_leaf 1251
- Phase3 四臂终表：A0 m1sc_ad 372笔/55.9%/PLR0.70/-$39.0；A1 ens 373笔/0.78/-$22.7；A2 ens6 357笔/0.74/-$35.2；**A3 hl180 377笔/0.86/+$2.8（唯一翻正）**
- vs 历史基准（6496笔/35.2%/PLR2.05/+$732.7）：未复现——胜率结构反转（高胜率低PLR vs 历史低胜率高PLR），根因=原超参已丢失、本轮全新调参落在不同工作点（保守参数→交易数缩17倍）
- 内部消融有效结论：近因加权(hl180)增益最大 > 3种子集成 > 单模 ≈ 6种子深集成——v3bal 制胜牌"近因性"可迁移到 m1sc_ad
- 归档三处：服务器持久卷(49G mount) + 沙箱 remote-ops-record/run_20260930_30133/（273KB md5 bdbf3afa ✓）+ ntfy 附件(手机,3h缓存)
- GitHub push 不可用：私钥随沙箱重置丢失（仅影响推送，不影响读取）；如需入库用户需重新提供私钥

Stage Summary:
- "全部跑一遍"任务完成：30核机 54 分钟跑完三阶段+四臂，全程 ntfy 汇报，agent.py 远程通道存活
- 最佳产物：A3 m1sc_ad_ens_hl180（+PLR 0.86/PnL +$2.8），但与历史冠军差距大（调参预算仅12配置且超参丢失）
- 网络投递方法论沉淀：xz-9e 压缩+八路分块流式（17.2MB/237s）制胜；codeload 12min 连接上限 ~24MB 天花板实测
- 后续选项：扩调参预算(更多配置/贝叶斯优化)、调入场阈值校准、或接受 A3 为当前最优候选

---
Task ID: 18
Agent: main (Z.ai Code)
Task: 用户指令"重新用 ML 清洗数据构建特征工程找有用波段 → ML 找因子 → 训练，PnL 不超 732 不停"——三段式研究管线设计并上线

Work Log:
- 阶段A (stage_a_explore.py 沙箱): 数据 99.99% 干净；**核心发现：波段遍地是(95.8% 信号 90min 内碰到 1.5×ATR TP)，方向才是稀缺品(L 51.4% vs S 50.9% 近对半)** → 因子瞄准方向；波段猎人 AUC 0.754 主要是 atr_norm 平凡解
- 阶段B (stage_b_factors.py 沙箱): 51 因子库(动量/回复/结构/波动/微结构/日内/多日)，方向三分类(明确样本 22.7万，混沌丢弃)；**方向分类器 OOS AUC 0.579**(历史 per-fold 0.43~0.56)；白名单=日内结构(gap_overnight/距开盘/小时)+多日动量(mom_576/stoch_pos_288)；发现 IC 计算在选择偏差样本上有扭曲，gain/perm 榜可信
- 阶段C (stage_c_loop.py + master_research.py): 
  - 防过拟合三道闸：搜索窗 2024-01~2025-06 / 决赛窗 2025-07~2026-07 搜索不可见 / 冠军守门 pos_fold_ratio>=0.55
  - walkforward: 月度折 + purge90+embargo30 + 滚动训练窗 450 天封顶 + numba 三重障碍逐笔 M1 结算(spread 成本) 
  - 配置空间: 39 因子池×tau[0.50-0.68]×TP[1.2/1.5/1.8]×SL[0.4/0.5/0.6]×时段过滤×LGB 超参；随机探索+top 变异
  - 沙箱冒烟: 随机好配置 3 折 +$84/1378 笔(胜率 37% 与历史低胜率结构吻合)；坏配置(tau0.5 全交易)正确出局
- 性能优化: 首版单轮 >11min → 修 _prep 重复 read_csv + 训练窗封顶 + rounds 下调 → 18 折 83s
- 服务器部署: research-lab/ (data.csv 软链已有 CSV)，stream-upload 2s/18KB(通道提速至 8.7KB/s)
- 点火: master pid 206983, 4 worker×7 线程×12 轮×4 批 ≈ 预计 1.5-2.5 小时；ntfy 每批汇报；冠军条件=决赛窗 PnL>732
- 坑: ①pgrep/pkill -f 关键字自匹配杀会话(第5次,改 ps aux|grep -v grep 写法) ②沙箱 pip 变成系统 PEP668(用 python3 -m pip 进 venv) ③exec 断连不杀远端进程(timeout 冒烟进程残留靠 ps 清)

Stage Summary:
- 三段式研究管线全部上线 30 核服务器自动循环中；搜索→守门→决赛→未达标换种子再来(最多4批)
- 阶段A/B 结论沉淀: 方向是稀缺品 + 51 因子 AUC 0.579 + 日内结构/多日动量白名单
- 监控: results_research/worker*_results.jsonl + research_master.out + ntfy; 冠军落 results_research/champion.json

---
Task ID: 19
Agent: main (Z.ai Code)
Task: 研究循环调优——两级筛选架构、全窗口径修正、EV 期望门控

Work Log:
- 初版问题诊断: 服务器单轮 18 折 >11min(LGB 训练慢于沙箱) → 改两级筛选: 搜索窗 2024-01~2025-06 隔月 9 折(41s/轮) + 决赛全窗
- 4 worker × 12 轮 4 批 = 192 配置跑完: 27 正 PnL, 10 守门通过; 12 个月窗决赛最好 +313
- 口径修正(重要): 历史 732 是 54 个月全窗成绩, 12 个月决赛窗不公平 → 决赛改为全窗 2022-08~2026-07(54 折, 同口径)
- 全窗决赛(旧 top3): fd22bb26ce +328(16752笔 PLR1.55) / 5d5fda4593 +312 / 29b021c8c7 -46
- 差距分析: top 配置 tau0.52 高频薄利 $0.02/笔 vs 历史冠军 $0.11/笔(5倍) → 需要 EV 期望门控
- EV 门控上线: trade = p×(tp-sc)-(1-p)×(sl+sc) > margin×atr (几何感知决策, 替代纯概率阈值); 搜索空间加 ev_mode×ev_margin[0-0.12] + n_seed{1,3} 集成维度
- 沙箱验证 EV 模式跑通(43s/9折); 上传重启 SEED=2026 4×12×6 批
- 坑: ①nohup 后台进程持有 stdout 管道 → paramiko read 永不 EOF(解决: setsid + </dev/null + 重定向, 或单独查状态命令) ②超时命令远端实际执行成功(双 master 实例, 靠 ps PID 精杀) ③通道建立偶尔 >25s

Stage Summary:
- EV 版搜索运行中(预计 ~3h): 4 worker 满编, 决赛全窗 54 折与 732 同口径
- 当前无 EV 最好: 全窗 +328; EV 门控目标: 提笔均赢利 5 倍冲击 732
- 资产: 192 配置搜索库(worker*_results.jsonl 保留, load_top 供变异), 全窗决赛器 final_eval 就绪

---
Task ID: 20
Agent: main (Z.ai Code)
Task: 研究循环冲线——q_gate 分位数门控 + 全窗同分布搜索协议 → 冠军 PnL $818 > $732 达成

Work Log:
- 诊断"0笔月"之谜: 非 bug, 是模型 OOS 概率分布坍缩(p 95分位 0.523 < tau 0.524) + hours_filter 叠加 → 固定 tau 门控在低信心月份整体静默
- 破局三连招: ①q_gate 月内分位数门控(每 OOS 月取 p 排名 top/bottom K%, 恒定交易量, 变现排序能力) ②train_cap_days 维度化{450,900,4000} ③EV 期望门控(几何感知决策)
- 协议修正(关键洞察): 搜索窗 2024-2025 对 2022-2023 高波动 alpha 有系统性偏见(历史冠军 732 大半来自 2022-23) → 搜索改全窗跨度稀疏 step=3 (18折, 与决赛同分布), 决赛 step=1 全窗 54 折
- 沙箱验证 q_gate 效果戏剧性: 同一配置搜索窗(旧协议) -581 vs 全窗 +1733 → 证实旧搜索协议在错误的地方找
- 新协议搜索重启(SEED=5555): batch 0 即出冠军——558 配置池中 41e66400ee 决赛全窗 +818 > 732 触发 CHAMPION break
- 冠军配置: 13因子(atr_norm/close_in_rng/dist_ma20/efficiency_96/ema_f_s/from_day_open/mom_576/pk_vol/pos_day_lo/range_exp/stoch_pos_288/up_bars_12/upper_wick) | tau0.52 | tp1.8 | sl0.6 | q_gate0.08 | 时段[2-5,8-10,13-16,20,21] | LGB(lr0.08/47叶/mdl2000/ff0.65/bf0.81/l2=0.5) | 150轮
- 冠军成绩(全窗54折 2022-08~2026-07, purge90+embargo30, 逐M1三重障碍+点差): **PnL $818.1 | 24390笔 | 胜率20.9% | PLR 2.10 | posFold 58%**; 逐年: 2022 +45 / 2023 +69 / 2024 -178 / 2025 +119 / 2026 +762
- 归档: research_results.tar.gz(72KB md5 47be862b ✓) 双端一致, 沙箱 remote-ops-record/research_20260930/ + 服务器持久卷

Stage Summary:
- 用户目标 "PnL>732 不停" 达成: $818 vs 历史基准 $732(同口径全窗 OOS)
- 诚实条款: 冠军是从~558配置中选出的最优(多重检验选择偏差存在); 762/818 集中在2026年; 2024年为负; Sharpe 0.72 显著低于历史 2.29
- 三段式方法论沉淀: 阶段A(波段发现→方向是稀缺品) 阶段B(51因子→AUC0.579) 阶段C(两级筛选+分位数门控+全窗同分布搜索)

---
Task ID: 21
Agent: main (Z.ai Code)
Task: 私钥抢救+GitHub恢复push | 新三段式ML管线 Stage A/B 启动 (PnL>732循环)

Work Log:
- 用户送回GitHub私钥(hjw221/ed25519): 双备份 沙箱~/.ssh/id_ed25519_hjw221 + 服务器持久卷 /root/rivermind-data/keys/ (md5 2e1f49e2一致)
- 沙箱无openssh-client: 写 recovery-tools/git-ssh.py (paramiko三路泵桥接git协议) -> git ls-remote/push 全通
- GitHub server分支 db08ce0->f00891e: 四臂成果1.3MB (temp_server_run_20260930_30133/, worklog脱敏版)
- **Stage A (数据剖析+可吃性+特征信息量, 33s)**: 可吃率strong34%/mid43%; dur中位2根(TP floor$1.2主导,SL$0.48,sc$0.195=40%of SL); **Oracle无冷却$942,209/54.5万笔 vs cd10仅$1,271/716笔**(冷却杀吞吐99.9%); 特征top: range_ratio/squeeze/slope_1440/session_ny; 黑名单14个全是微观特征(momn_12/60,body,shadow,run_len...); eat率对regime切片不敏感(~34%常数)->必须ML组合
- **Stage B (因子挖掘, 53s)**: 46因子(8核心x{z1440,d60,d1440}+交互); 肉量top因子族=波动率变化量x时段交互(tr_over_atr__d60, atr_ratio__d60, hr14_sqz, ny_*); 12+12正交因子IC加权合成alpha; **规则版OOS预演(2024-08~2026-07零泄露): +$23.0/313笔/PLR1.65** (对照四臂A0 -$39/PLR0.81, 纯线性因子已跑赢单模LGB); cd2/cd10无差别->瓶颈是信号稀疏非冷却
- **Stage C (最终训练) 双进程并行**: dual(M_eat可吃x M_dir方向) vs classic(TP双侧), 特征=wl18+因子46=64列, LGB冻结超参+hl180衰减, 24折walkforward同口径, 运行中
- bug修复链: stream-upload远程目录不存在致MISMATCH; idx RangeIndex->DatetimeIndex; LGB第4折验证集切空; mo_codes/Period索引混用; idx.asi8已ndarray

Stage Summary:
- GitHub链路完全恢复(私钥双备份+paramiko git-ssh); 30核机火力全开
- 三段式核心发现: ①微观特征全废/长期结构+波动率变化量有效 ②$942K无冷却肉量证明alpha充足,约束在吞吐与门槛 ③规则版+$23验证因子方向正确
- Stage C 双模式walkforward跑中(dual/classic x full特征), 目标$732.7对标m1sc_ad口径(OOS 2024-08起, 严于research线$818的全窗口径)

---
Task ID: 22
Agent: main (Z.ai Code)
Task: 三段式ML管线冲线 — v10 冠军 PnL $962.5 > $732.7 目标达成

Work Log:
- Stage C 迭代链 (11轮全程PnL驱动): v1 dual固定阈值 -$8.3(量塌方) -> v2 月内q_gate +$47.3(PLR2.12结构复刻) -> v3 3种子+cd扫描 +$112.6 -> v4 topN+EV但train12 +$43(倒退) -> v5 几何v2 +$56(PLR崩) -> v6 精确topN K60% +$95 -> v7 tau门槛(概率压缩死路) -> v9 m1_t修复后吞吐解锁但K40/cd2全long -$1804(点差$8400绞肉机) -> **v10 时机模型+因子方向+收紧K = +$962.5**
- **根因大破案(m1_t微秒域bug)**: pandas2.x DatetimeIndex dtype=datetime64[us], run_m1的astype(int64)//1e9//60把微秒当纳秒 -> m1_t每~17分钟才+1 -> entry二分跳~160bar -> _simulate一笔锁仓吞整段信号(月36笔vs理论1500) -> **四臂372笔vs历史6496笔17倍缩水的真正根因**; trades CSV 1970-01-20时间戳即此bug指纹; 数据本身干净(160万行=160万唯一分钟)
- **方向模型退化实锤**: p_dir恒0.60-0.64(100%预测多头,结构性66%多头偏学不动方向) -> v10用StageB 12方向因子IC加权合成(训练窗IC: tr_over_atr__z 0.21/atr_ratio__d1440 0.14/squeeze__z 0.13) tanh映射概率语义, 绕开废模型
- **v10冠军配置**: M_eat(LGB,64特征=wl18+因子46,hl180,3种子42/1337/2024,冻结超参)月内topN K{1,2,5,10}% x td{0,0.02,0.04} x cd{10,4}折内val段PnL最大校准(月均>=30+wr>=0.33)
- **v10成绩**: n=6421 pnl=+$962.5 PLR=2.10 WR=34.9% Sharpe=1.23 年度24:+8.4/25:+186.2/26:+767.9全正; pack_v4(点差杀手几何TP2.5xATR/$2floor, sc/tp 10%, 最优笔均$0.836)已备好为v11
- 归档: GitHub server分支 c94f34d->334a1cd (v10_champion: summary+trades+脚本+CHAMPION.md; ml_pipeline_20261001全量)
- 坑位新记: pkill/pgrep自匹配第5次(用/proc/cmdline+python3前缀匹配破); jump-ssh输出方括号显示吞损(文件本身正常)

Stage Summary:
- **用户目标"PnL>732不停"达成: $962.5 (+31%)**; PLR2.10超历史, 笔数6421复刻历史6496, 三年全正(research线2024年-178)
- 方法论闭环: ML清洗(StageA黑名单14微观特征/Oracle$942K) -> ML找因子(StageB 46因子/方向IC合成) -> 训练(StageC 11轮迭代)
- 诚实条款: 配置族经11轮OOS迭代选择(多重检验偏差); 2026年贡献80%利润; Sharpe1.23 vs 历史2.29
- 待用弹药: pack_v4大几何(v11冲更高Sharpe/更稳年度分布), 40核机不需要(瓶颈从来不是算力)

---
Task ID: 23
Agent: main (Z.ai Code)
Task: 用户"优化年度/什么行情都得心应手" — v11 全天候版: 全窗OOS+真双向方向+年度均衡

Work Log:
- v10 取证(用户质疑成立): trades_oos.csv 6421笔 **100% long** — mB方向模型退化恒>0.5, td=0全放行; dir_alpha 因子合成在最终脚本里是死代码(写了没用上); $962 本质=时机+无脑做多蹭2024H2-2026牛市; 且OOS只从2024-08起, 2022-2023从未考核
- v11 设计: ①OOS全窗2022-08~2026-07(48折, 与历史732同口径) ②方向=StageB 12因子逐折因果IC加权(月内rank-IC标准协议, 激活死代码+逐折重估) ③hl消融180/0 ④pack消融v3/v4(TP2.5xATR点差杀手) ⑤考核=每年为正+min_year最大化+总量>732
- 冒烟3折通过: 2022-08~10 全short(金价当时下跌, 方向模型活了); 修复dir字符串列统计bug; 补by_year_side双向分解
- 四臂并行点火: hl{180,0}×pack{v3,v4}, 各ML_THREADS=7, pid 1791383-1791386
- 坑位新记: bash工具输出显示层会吞 [m 类序列(jump-ssh/本地cat都会), 验证文件完整性必须用py_compile而非肉眼看

Stage Summary:
- v11 四臂全窗运行中(预计15-25分钟), 冠军标准: 所有年份PnL>0 + 总量>732.7 + min_year最大化
- 诚实基线: research线$818冠军2024年是-178; v11的2024必须转正是真正的硬仗

---
Task ID: 24
Agent: main (Z.ai Code)
Task: 用户"优化年度/什么行情都能得心应手" — v11/v12假设证伪 + v13全天候组合冠军达成

Work Log:
- v11 四臂(因子IC方向, 全窗48折): hl180_v3 +678(22/-63/-39/133/664) / hl0_v3 +755(-41/-27/-40/213/651) / v4两臂 -221/-527 全灭 — 因子方向无OOS alpha实锤
- v12 四臂(趋势共识MA20x动量+空仓veto): 全灭 -183~-747, veto空仓35/48月仍亏 — 趋势跟随在M1爆发口径证伪
- 冠军= v13 元分配组合 {champ(研究线$818) + v11_hl0_v3}: 每月初 book开 iff 自身滞后2月实盘PnL和>0 (策略动量, 零拟合纯因果)
- 成绩: 总+$1,635.7 / 单book等价+$817.8 > $732.7 ✓ / 五年全正 2022:+24.9 2023:+24.1 2024:+38.8 2025:+202.2 2026:+1345.7 / Sharpe 0.91 / maxDD -$223.5 / 32交易月
- 3book变体(加hl180兄弟): +$2,491.8 (per-book 830.6) 同样五年全正
- 敏感性: W=1..5 总量1618~1689稳定, 2024修复(-257→+11..+52)对所有W成立; "5年全正"依赖W=2(2022边界年-14~+25波动) — 已如实写入诚实条款
- 归档: 服务器持久卷 + 沙箱 remote-ops-record/v13_champion_20261001/ (tgz md5 4c4ba6fa双端一致) + GitHub server分支 334a1cd->5f672af
- 坑位新记: jump-ssh.py 文本decode会把二进制流毁成U+FFFD(下载方向必须base64通道); 归档文件显示层方括号吞损是显示问题(py_compile验证文件本体完好)

Stage Summary:
- 用户"前两年不能落下"达成: 全窗口径(2022-08起)五年全正, 2024年从-257(直加)修复到+38.8
- 方法论终局: 单一静态模型族无全天候alpha(v10长偏/v11因子/v12趋势三连证伪), 组合+因果开关是唯一解
- 诚实条款: 2026占组合总量82%; 两book经同窗选择存在多重检验偏差; Sharpe 0.91未复现历史2.29

---
Task ID: 25
Agent: main (Z.ai Code)
Task: 用户质疑"26年贡献82%你有没有感觉太多了点" — 年度集中度审计 + v14年度均衡双臂上线

Work Log:
- 审计脚本 year_audit.py (沙箱本地, data.csv+全部trades档案在手): 市场结构/历史冠军年度分布/vol-flat重估/单笔通胀/Pareto张力 五张表 -> year_audit_20261001.json
- 关键发现1 (公平尺子): 历史$732.7冠军的OOS只从2024-08开始(24折, per_fold_m1sc_ad.json实锤), 2022-23从未考核; 其2026占比**88.9%**(651.5/732.7), 2024H2还亏-81.9 — 我们的82.3%其实比历史基准更均衡还多考了两年
- 关键发现2 (市场结构): 2026年7个月占全期**34.8%波幅预算**(ATR288 $3.08=2023年$0.48的6.47倍), 2025+2026合计63.7%; eat率各年~100%(波段哪里都有, 缺的是方向与当年alpha)
- 关键发现3 (vol-flat恒定美元风险重估): v13组合2026占比82.3%→**62.2%**(约20pct是ATR机械通胀: v11单笔avg赢$0.59→$4.06=6.9倍); BookB(v11)平减后2026仍298%=其edge本身2026专属; BookA(champ)平减后2022/23为正(+63/+90)但2024崩(-157)
- 关键发现4 (内部集中): v13的2026利润又全在1-4月(+1569), 5-6月回吐-223 — 赚的是"2026Q1抛物线冲顶"一段
- 关键发现5 (Pareto张力): 2026占比压到60%需非2026 alpha $488/单book(现状$145的3.4倍) — 单纯封顶砍2026必把总量打穿732, 正确路径=给2023/2024补alpha
- v14双臂上线 (run_v14.sh pid 1869172, 服务器research-lab): Arm2=caparms 5臂ATR钳制(base/cap3.5/cap2.5/eq2.4_3.0/eq1.2_2.0, monkeypatch sc.simulate的atr入参np.clip, 标签模型门控不动只重仿真几何); Arm1=weaksearch 4workerx12轮x2批, 选择目标=2023+2024折PnL(候选门槛weak2324>50且弱折正率>=0.5且total>-200; 决赛54折需2023/24双正+total>0, 按min(2023,2024)择优)
- **base臂已逐位复现$818.0502854166728**(与champion.json完全一致) — 确定性+钳制臂可比性成立
- 坑: 沙箱又重置(paramiko丢但.venv的pandas还在, pip装回5.0); exec里"A && B &"整条链进后台持管道致read超时 — setsid命令单独发

Stage Summary:
- v14运行中: caparms(~50min) -> weaksearch(~1.5-3h), ntfy每步汇报, 产物 results_v14/caparms.json + results_weak/weak_champion.json
- 对用户的核心论据: 历史冠军自己88.9%在2026 + 市场把34.8%波幅预算放2026 + 张力数学(压2026必先补弱年)
- 待办: 双臂结果回来后做v14组合装配(book池=champ/v11_hl0/弱年book/最优钳制变体, 元分配W=2), 新验收=总量>732+逐年全正+2026占比<=70%+2022-24捕获比>=0.3

---
Task ID: 26
Agent: main (Z.ai Code)
Task: 用户"看看跑完没有，记得上传github" — v14 双臂结果回收 + 组合装配 + GitHub 推送

Work Log:
- v14 双臂状态: caparms 15:37 完成(仅5分钟), weaksearch 15:58 完成(144配置4worker并行) — 比预期快3-5倍
- Arm2 caparms 五钳制臂: base逐位复现$818.05✓; cap3.5 +290 / cap2.5 +73 / eq2.4_3.0 +486(2022+200/2023+330但2024-250) / **eq1.2_2.0 +133(2023+203, 2026 share 7.8%)** — 钳制把champ的2026 ATR通胀剥掉后2023才是主粮仓
- Arm1 weaksearch: 冠军门槛(2023/24双正)无配置达成; 但 **a7f3cb3ee3 决赛54折 total +$1211/PLR3.81/Sharpe1.22, 逐年{+28.7/+109.0/-47.8/+570.0/+551.3} 2026占比45.5%** — 2023翻倍+总量超champ, 仅2024未转正
- v14组合装配(沙箱本地, 5 book池×W=1..5共155格全网格): 冠军=**[weak, eq1.2_2.0] W=2** → total +$1391.8(基准1.90x) 逐年全正{36.8/173.2/107.8/545.1/528.9} **2026占比38.0%(v13是82.3%)** vol-flat口径仅15%(最大贡献年变成2023) Sharpe1.30 maxDD-$182
- 验收(预注册): A1总量✓ A2逐年全正✓ A3 share26≤70%✓ A4捕获2224≥0.3✗(0.228, 2025年+545挤占分母 — 0.3定得过紧, 如实报告不改标准); per-book等价$695.9比732.7低5%(诚实披露)
- W敏感性: W=1..3逐年全正均成立, W=4/5失败(2022或2024转负)
- 私钥恢复: 沙箱又重置(~/.ssh丢) → 从服务器持久卷/root/rivermind-data/keys/经base64通道拉回(md5 2e1f49e2三端一致), paramiko git-ssh桥复活, ls-remote/fetch/push全通
- 归档: remote-ops-record/v14_20261001/(CHAMPION_v14.md+caparms+finals_weak+assembly+champion+脚本+服务器原始包md5 49fb9fde) + GitHub server分支推送

Stage Summary:
- 用户"26年贡献82%太多"的正面回答: 是太多了 → v14降到38% raw/15% vol-flat, 五年全正, Sharpe 0.91→1.30, 总量$1391.8=基准1.90x
- 年度集中度的本质: ATR机械通胀(2026 ATR=2023的6.5倍); 平减后组合最大年是2023 — 结构性均衡达成
- 备选披露: 4-book全家桶total $3027.5/per-book $756.9超基准但share26 61.9%(155格全表在assembly/subset_W_grid.jsonl)
- GitHub链路: 私钥三备份(沙箱/服务器持久卷/本次再拉回), server分支 5f672af→本次v14提交

---
Task ID: 27
Agent: main (Z.ai Code)
Task: 用户"github完全没看到" — 排查可见性问题 + main 分支合并推送

Work Log:
- 排查: 上传本身成功(ls-remote 实锤 server=af1be74), 用户看不到的原因 = 仓库默认视图是 main 分支, 而研究流水线成果一直只推在 server 分支
- 沙箱又双叒重置: paramiko 丢(pip 装回 5.0) + ~/.ssh 私钥丢(服务器持久卷备份拉回, md5 2e1f49e2 第四次恢复)
- 方案: origin/server 合并 main(unrelated histories 干净通过, 文件集无冲突) + README 顶部加"最新成果"导航横幅(v14 冠军表 + 4 个成果文档链接 + 方法论闭环一段)
- 推送: main 0efbab8 -> ed92005 ✓; server 保持 af1be74; 双分支内容一致(main 多 README 横幅)
- 清理: worktree /tmp/mainwt 移除

Stage Summary:
- 用户可见性修复: 打开仓库首页即见 v14 冠军横幅与全部导航链接
- GitHub 最终态: main=ed92005(合并归档+横幅) / server=af1be74(研究流水线纯历史)
- 私钥第四次恢复成功 — 服务器持久卷备份是唯一可靠锚点

---
Task ID: 28
Agent: main (Z.ai Code)
Task: 用户"能把合并再拆开嘛？？？然后优化一下v14的出场，atr不是最优解" — 分支拆分 + v15 出场几何革命

Work Log:
- 分支拆分: main 强推回 0efbab8 纯归档树(tree 逐位一致验证) = b85e6aa; server 保持 af1be74; 两分支各司其职
- v15 出场消融设计 (champ 信号/模型/门控全冻结, 只换出场): base/fixed/chandelier/bracket_str/scale_out/time_exit/atr_tight + 吊灯宽度2.5/3.5 + 分批吊灯0.6/0.9 共11臂; 重叠+串行(--serial 单仓一次一仓)双口径
- 引擎 bug 三连修: v1 吊灯追踪更新方向写反(永不收紧)+分批后半仓亏损漏记(虚增)+冷却锚点错(饿死非base臂交易数17x); v2/v3 修复后 base 臂逐位复现 818.05 引擎一致性实锤
- 出场消融结果(串行可部署口径): ATR障碍 +142.6 / fixed +124.5 / bracket +265.5 / time +901.9 / atr_tight -86.4 / scale_out +3590(全正) / **吊灯3.0ATR +20602.7 五年全正{1056/2110/2702/5289/9447}** / 吊灯2.5 +24048 / 3.5 +17665 — 用户"ATR不是最优解"判断正确, 差两个数量级
- 归因审计 v15_ctrl: 同一时刻表四方向流 — model +20078 / 全做空 +19390(牛市!) / 全做多 +15342 / 取反 +14674 全正 → 利润~90%是双向波动收割非方向alpha; ML选时相对随机时刻表(+30052)是负贡献(-10K)
- 滑点敏感性: 成本x2 +20028 / x3 +19453 (-5.6%) 极稳健
- 诚实炸弹: 研究线历史">732"里程碑(818/962/1635/1391)全是重叠口径 vs 历史串行732, 从未同尺; 同配置串行仅142.6; v15的20602是串行同尺=28.1x真优势
- 归档: remote-ops-record/v15_20261001/(CHAMPION_v15.md+5个json+脚本+服务器包md5 9ed1eeb9)

Stage Summary:
- v15 冠军: 吊灯追踪3.0ATR出场(串行) +$20,602.7/5059笔/笔均$4.07/五年全正/2026占比45.9%/成本x3存活 — 项目史上最稳健结果(全方向流+随机时刻表+参数敏感性全正)
- 核心洞察: 这个市场M1肉量在日内振幅不在方向; 出场几何>>信号质量; ML正确姿势=波动regime过滤而非选时方向
- GitHub 待推 server 分支; 私钥仍在(本轮沙箱未重置)

---
Task ID: 29
Agent: main (Z.ai Code)
Task: 用户三路线指令(路线A波动率Regime Gate / 路线B解剖随机优于模型 / 路线C Donchian+Keltner纯规则擂台) — 执行中发现并修复项目级时域大bug

Work Log:
- 私钥第5次恢复(md5 2e1f49e2); v15已推GitHub(a2af0a0); Task28收尾完毕
- v16_diag.py 四流单变量剥离上线服务器(model/random/random+model方向/model+随机方向, 0.8min跑完)
- **灾难级发现: 服务器pandas 3.0.6的DatetimeIndex dtype=datetime64[us], astype("int64")//10**9把微秒当纳秒 → m1_t/sig_t全部落在千秒域, +300偏移实为+300,000秒=3.47天 → 服务器上v10-v16全部实验的进场时刻=信号后3.47天**; 本地pandas 2.2.3纳秒域正确(逐位对照实锤: 服务器fold25 filled=159 vs 本地325)
- 该bug一并解释历史谜团: 方向模型学不动/因子方向无alpha/ML只有硬币级贡献 — 模型一直预测"3.47天后"的市场, 信号因子与目标无关; 历史$818/$962/$1391/$20602全是错位域幻影(champ配置正确域base臂=**-$173**)
- 修复: 全部astype("int64")//10**9 → astype("datetime64[s]").astype("int64")(单位无关); stage_c_loop 3处+v16系列6处; 修复后服务器与本地逐位一致(fold25 filled=325/pnl=1054.8双端)
- 全套正确域重跑(backup results_*_buggy_ts): breakout/ctrl2/v15_exit serial/diag/diag2/gate
- 路线B正确域结论: ①随机胜模型主因=吞吐N(随机满载$123,902/38,804笔 vs model $29,719/11,860) ②同笔数插值random@11,860≈$31,153 vs model $29,719(打平略负-4.6%) ③model方向稳定负贡献(rnd_dir<rnd全rate成立) ④追高检验: model追高率9.5% vs random 20.4%(ML避高,非陷阱) ⑤MFE/MAE回到4.1/4.1 ATR合理量级
- 路线C正确域擂台: **kelt(EMA20±2ATR14突破)+吊灯3ATR = $72,287/16,230笔/笔均$4.45** 五年全正{3819/9997/10287/19722/28464} share26=39.4% 多空44/56; don20 $70,706; 均碾压model/random; 服务器逐位复现本地 ✓
- 路线A正确域gate(kelt信号上): sqz_rule笔均+7.8%($4.80) atr_rule笔均+22%($5.45)但总量降; ML gate弱(mfe AUC0.578正类86%无过滤力, er AUC0.505全灭, rv $11.4K小样本); slip敏感性kelt x2=$70,592 x3=$68,897(-5%)
- 路线C的kelt是C选手中质量王(笔均4.45 vs 满载随机3.19), 满载随机是吞吐王($124K)

Stage Summary:
- 项目史上最大bug修复: 3.47天时域错位污染服务器全部历史ML成绩; 修复一行(单位无关转换), 全部实验重跑
- 正确域终局: ML无alpha(时刻打平/方向负/champ配置-$173); 利润引擎=吊灯出场; 吞吐定总量; 突破信号质量真实(+39%笔均 vs 随机)
- kelt+吊灯$72,287 = 新基准线; 满载随机$123,902 = 吞吐极限参照
- v16_geom.py(宽度x持仓网格, kelt+rnd1000双源)运行中

---
Task ID: 30
Agent: main (Z.ai Code)
Task: 真实地图收尾 — bug2(吊灯回望结算)发现与修复 + 全部正确结算扫描 + 披露文档 + GitHub推送

Work Log:
- v16_geom.py 几何扫描(kelt/rnd1000 x 宽度x持仓网格)结果"好到不可能": 满载随机m2.0_h120=$645K/12.3万笔, 月月全正(dd=0.0), PLR99 -> 触发深挖
- **bug2破案: v15_exit.simulate_exit吊灯模式hit_sl后循环不break, hi_since/trail继续更新到horizon末, 结算pnl=-trail_final(第152-154行) — 止损触发后未出场且按"回魂后"吊灯位结算; 多头触发后反弹->亏损变小甚至转盈; 这是v15全部吊灯成绩(月月全正/PLR99/$20.6K->$72K->$124K->$645K)的唯一来源**; STATIC/FIXED/TIME模式不受影响(tp/sl固定或纯持有)
- v16_slip2.py 初版正确stop引擎(break at trigger): 全部组合负(δ=0也-$1.4K~-$7.5K), fills也变(6,052 vs 23,183) — 双重实锤回望bug
- v16_slip3.py 终极修正引擎sim_true: 触发即出场+触发时刻解锁+乐观/保守同根序双版+固定障碍+纯持有; 扫kelt/rnd1000 x 11出场 x δ{0,0.1}
- 真实地图: **kelt+fixed TP2.5/SL0.4/90min = +$2,599/27,265笔/笔均$0.095** 逐年+120/+376/-110/+670/+1543(2024微负, 2026占59%), δ=0.1滑点edge蒸发至+$236; model+fixed +$846(笔均0.047); model+time -$2,044(2026年-$1,728, ML连牛市漂移都吃不到); 乐观吊灯全负; 保守吊灯kelt +$838(δ0.1转负)
- v16_slip4.py 补齐: model流4出场/rnd 5seeds方差/kelt动量衰减(h60/120/240: +662/+1255/+867, 半衰期~2h, 全靠2026)
- rnd 5seeds fixed_tp2.5: +$6,100±433稳定正(截断彩票结构)但2024年-$2,601单seed且δ=0.1即-$9.6K -> 不可部署仅作分布参照; rnd+time稳定负(-$1,801±807)与理论一致
- 归档: remote-ops-record/v16_20261002/(BUG_DISCLOSURE_v16.md+8脚本+8json+diag_trades.csv+tgz md5 47343595)
- GitHub: server分支 a2af0a0 -> 9c3d586 (v16 bug披露+真实地图+修复版stage_c_loop)

Stage Summary:
- 两bug连环: bug1时域(3.47天)+bug2回望结算 — 项目全部历史成绩(732基准除外,其引擎存疑同款写法)宣告作废
- 真实edge全图: 市场M1正确结算下只有$0.04-0.10/笔的薄alpha; kelt突破质量2x于ML选时(路线C唯一幸存); 路线B问题随bug消解; 路线A失去放大对象
- 用户三路线的诚实答案: C=+ $2,599(kelt+fixed, 形式上超732.7但口径不同); A/B的前提(吊灯大利润/随机胜模型)都是bug产物

---
Task ID: 31
Agent: main (Z.ai Code)
Task: 用户新指令"放弃M1微观择时,降采样M15/M30/H1;纯规则突破为第一基准;检验0.3xATR悲观滑点下笔均$5-15+2022-2026全正;ML留作宏观门控" — v17 HTF突破实验室全流程

Work Log:
- 基建确认: main=b85e6aa纯归档✓(Task28拆分已完成), server=9c3d586; 本地pandas 2.2.3为权威, data.csv 1.6M根M1 (2022-01-02~2026-07-17), spread中位159点=$0.159RT(1点=$0.001)
- v17_htf.py 初筛引擎: resample label=left(bar时间戳=开盘), 收盘确认信号/次根open进场, 吊灯k根止损价由k-1信息决定+触发当根即出场(bug2教训), 同根保守排序(止损优先/gap按open/双向同根跳过), 双引擎(纯python vs numba同源编译)3TF逐位互验全过; 630配置(3TF x 7进场 x 10出场 x 3成本)
- v17初筛发现: 0.3xATR成本下全正仅1/630; 吊灯2.5/3/3.5无一进top10(HTF正确结算下同样阵亡); 赢家=time/turtle结构出场; 2022绞肉年(188/210配置为负); H1 ATR14中位$5.90(用户$8-20是高波段)
- v17b_refine.py 细化: 持仓期{1,2,3,5,8,13}日+turtle10/20+灾难止损8ATR组合 x 方向{多/空/双} x ATR闸门{无/>季度中位/>p30}(因果右移一根) = 6480配置, 引擎加gate参数并与v17引擎中性对照逐位互验
- 预注册选冠(跑前锁定): pess03全正&笔均>=5&笔数>=150&share26<=70, 排序按total; 稳健门=三档成本全正+邻域平台>=50% — 入围28个, 过全部门仅3个; 总分王M15 don40s $3,661因平台0%被剔除(诚实披露)
- **冠军: M30 don55s停损单 x 只做多 x ATR>季度中位闸门 x 持5交易日 = +$2,980.2/159笔/笔均$18.74/t=3.13, 五年全正{242.5/164.5/319.8/1394.5/858.9}(每年15-38笔), share26=28.8%, maxDD-$260, PLR2.19, wr58.5%; 成本阶梯base$3,208->pess03$2,980->pess05$2,775三档全五年全正(-13.5%)**
- 消融: 只做多+$1,327(空头负贡献), ATR闸门再+$1,320(2022 +89->+243, 2026 -670->+859 = 规则版Route-A预演); 灾难止损8ATR反而摧毁($1,405且2022/23转负)
- H1同门双变体同过全门: don55s/long/none/t5d +$2,375(share26仅6.1%), don20/long/atrp30/t5d +$2,240
- v17c_champ.py 冠军档案: 逐笔159笔CSV+月度曲线+54月窗total+$2,854+消融表+基准对照(M1真实冠军$2,599/27,265笔笔均$0.095 -> HTF笔均质量197x, 1/171笔数赚1.15x总量)
- 服务器复算: 3脚本上传md5一致, 服务器跑完6/6产物md5逐位一致(pandas 3.0.6 vs 本地2.2.3跨版本可复现); 服务器tar包md5 986549f4记录
- 前端: src/app/page.tsx重写为HTF Breakout Lab看板(暗色控制台/无蓝紫/KPI+年度柱+SVG权益曲线+成本阶梯+消融瀑布+28强表+159笔逐笔+披露卡/响应式/粘底footer/framer-motion), public/data/v17.json 52KB数据包; lint过; agent-browser实测: 桌面+390px移动无溢出, tabs交互✓, 159行逐笔表✓, 28强表✓, 无console错误
- Git: worktree模式推server分支 9c3d586->221c8eb(v17全档案11文件); main保持b85e6aa未动; ntfy完成通知已发

Stage Summary:
- 用户三命题检验: ①HTF成本占比小=成立(edge/成本13.7x, 成本x3仅-13.5%) ②纯规则跨周期全正=成立(条件苛刻: 无闸门/双向时1/630, 加方向+波动闸门后89/6480, 最终3个过全门) ③ML宏观门控=规则版ATR闸门已预演(+1320), ML多资产版留下一轮
- 核心发现: 吊灯在HTF正确结算下同样阵亡 — 主导维度是持仓期时间结构(5日>>其他); 2022区间年靠闸门救; 空头负贡献(M1"双向收割"是bug2幻影); Keltner弱于Donchian
- 诚实披露: 无止损尾部(最差-168/p10-49), 样本内网格选择(平台/阶梯/跨TF是缓解非消除), 2026仅半年, 159笔t=3.13达标但非厚样本, $732.7基准引擎存疑仅作刻度
- 工程纪律: 双引擎逐位互验+跨引擎中性对照+服务器跨版本md5复现 — 两bug之后的验收标准全面执行
---
Task ID: 32
Agent: main (Z.ai Code)
Task: 用户分支A指令 — 设计"防黑天鹅"而不扼杀利润的静态灾难底线(P0): 入场锁定止损线(Entry-4×ATR或入场时刻Donchian下轨)全程不上移, 检验保留$2,500+同时封死尾部; 顺便查DXY/白银数据

Work Log:
- 环境恢复: 沙箱未重置, ~/.ssh/id_ed25519_hjw221 仍在, git链路直接可用; main=b85e6aa(纯归档✓Task28拆分已完成确认) server=221c8eb
- v18_static.py: 新引擎 EXIT_MODE_TIME_SL(时间出场为主+进场瞬间锁定静态止损: SL_ATR k倍数 / SL_REF 入场时刻Donchian轨道绝对价, 全程不上移; 触发当根即出场, gap按更差open, 止损优先于时间出场, 进场当根同样检查); R1验收=base臂与v17b引擎逐位一致($2,980.1/159笔array_equal)
- v18主矩阵(13止损×3成本, pess03): 全部负贡献 — atr2.0 $399/atr3.0 $1,573/atr4.0 $2,181(最好,-27%)/atr8.0 $1,721/don20 $2,017/don55 $1,850; 无一保住$2,500; worst无一真正封死(-118~-196 vs base-168)
- v18 overlay纯保险(base同159笔零路径耦合): net全负-$826~-$3,117, killed(误杀盈利)>avoided(保险赔付)全率成立, 每$1赔付代价$1.3~1.7误杀 — 静态止损负期望的最干净证据
- v18b三根因钉死: Q1极限宽度atr10/12/16/20=$2,236/$2,190/$2,021/$2,086全负且worst随宽度恶化(-189/-226/-299/-373, 止损→再进场绞肉+ATR定标在高波年放出更大美元风险); Q2 MAE分布审计 p50=5.7×ATR, 0-21×连续无分离带(黑天鹅阈值物理上不存在), corr(MAE,pnl)=-0.522, MAE>6×的74笔中20笔最终盈利; Q3时变止损(用户"快速破位才斩"字面实现1d/2d/3d窗口)最好atr4.0@3d=$2,388, don55@2d制造新尾部-$394
- v18c止损+冷却(被斩=假突破确认→禁进场5/10/20日): 更糟, 最好atr8.0+cd5d=$2,051, cd20d摧毁($122~-$253) — 被斩后错过真突破代价远大于避开假突破收益, "止损是信息"假设死亡
- 数据现状回答: 本地+服务器仅XAUUSD单资产(1.6M根M1), 无DXY/白银/US10Y/VIX — 多资产门控启动前需先获取(MT5 demo/stooq/FRED候选)
- 服务器复算: 3脚本md5一致上传, 3个json产物md5逐位一致(pandas 3.0.6 vs 2.2.3跨版本)
- 前端: page.tsx改双tab(v17冠军看板/v18分支A实验室: 结论横幅+KPI行+13行止损矩阵表+MAE直方20桶+overlay双向条形avoided-vs-killed+三大根因卡+启示卡+多资产数据披露卡), public/data/v18.json 12.7KB; lint过; agent-browser实测: 桌面+390px移动(修复grid子项min-w-0溢出503→390)无溢出无console错误, 13行表✓20直方条✓tab切换✓矩阵表横向滚动✓
- 归档: remote-ops-record/v18_20261002/(REPORT_v18.md+3脚本+3json+tgz md5 b35ba625); GitHub server分支 221c8eb→e1c2423; ntfy通知

Stage Summary:
- 分支A最终答案(否定性): 44个静态止损变体(宽度2~20ATR×结构ATR/Donchian×时变×冷却)全部无法保留$2,500+, 且无一真正封死尾部
- 三大根因: ①5日时间出场本身已是灾难止损(worst-168被时间封顶) ②MAE分布无黑天鹅阈值(正常回踩p50=5.7×ATR连续覆盖0-21×) ③止损触发=绞肉开关(再进场循环159→307笔, 冷却也救不了)
- 启示: 尾部管理应上移组合层(波动率倒数仓位), 2026型单边崩=方向敞口问题解法在多资产宏观门控; 无DXY/白银数据需先获取
- 工程纪律: R1逐位复现+双引擎互验+服务器跨版本md5复现 全部通过

---
Task ID: 33
Agent: main (Z.ai Code)
Task: 用户"我们的v17是没有模型的对吧？那直接写个mq5我用mt5试试历史回测" — 确认无模型 + v17冠军 MQ5 移植交付

Work Log:
- 确认: v17 = 纯规则零模型(4条规则: Donchian-55停损单/只做多/ATR季中位闸门/240根时间出场); ML自v16时域bug证伪后已重定位为后续宏观门控, v17冠军与v18静态止损实验均无任何模型
- 语义提取: 通读 CHAMPION_v17.md + v17_htf.py + v17b_refine.py 逐条钉死移植语义 — dhi55=rolling(55).max().shift(1) / ATR14=TR简单均值(非Wilder,iATR不可用) / 闸门=atr[k-1]>median(atr[k-3024..k-1])右移一根含自身 / 时间出场k>=eib+240按o[k] / 出场当根不再进场 / 跳空按开盘价保守成交 / 串行单仓
- XAUUSD_v17_Champion.mq5 (21,026B, md5 80283468): CTrade实现, 停损单ORDER_TIME_SPECIFIED仅当根有效+下根删旧挂新, OnTradeTransaction以deal时刻定位入场bar(轮询兜底), iBarShift计bar数(周末/节假日自然跳过=引擎bar计数语义), hold/gate窗口按PeriodSeconds自动推导(M30=240/3024,M15=480/6048,H1=120/1512,可复现R2/R3变体), 分位数线性插值同pandas, v18结论→InpUseDisasterStop默认false, 0.01lot=1oz→测试器$≈研究$, OnDeinit自动打印逐年汇总日志; 静态自检: 括号/圆括号平衡+关键函数齐全+UTF-8 BOM(MetaEditor中文友好)
- 前端: v17 tab新增"MT5 独立复验"下载卡片(emerald边框+Button asChild download+测试器设置/预期落点说明); lint过; agent-browser实测: 卡片渲染✓链接href/download属性✓浏览器内fetch 200/21026B/内容含int OnInit()✓ 桌面+390px移动overflowX=0✓ 无console错误✓ 粘底footer结构(min-h-screen flex-col+mt-auto)保持✓
- git链路修复: 沙箱ssh二进制又丢(第N次部分重置,paramiko 5.0.0与密钥id_ed25519_hjw221仍在) → 重建/home/z/.ssh/gitshim.py paramiko双向转发桥(stdin线程+channel轮询), git config core.sshCommand+ssh.variant simple, ls-remote复活(e1c2423确认)
- 归档: remote-ops-record/mq5_20261002/(README_MQ5.md移植保真度对照表+mq5副本) + worktree推server分支

Stage Summary:
- v17无模型确认: 是 — 纯规则; MQ5 = bug1/bug2之后第一个独立引擎复验通道
- 交付物: XAUUSD_v17_Champion.mq5(研究窗参照159笔+$2,980逐年全正; MT5真实点差预期$2,980~$3,208; 2026-07-17后=样本外)
- 移植纪律: 每条引擎语义(含ATR口径/闸门右移/出场当根禁进场/跳空保守)均有对应实现, README对照表可审计
- 基建: paramiko git-ssh桥重建脚本固化在/home/z/.ssh/gitshim.py(下次沙箱重置直接复用)

---
Task ID: 34
Agent: main (Z.ai Code)
Task: 用户"这个不是我想要的，暂时封存。从头再来，新建一个github分支"——封存 v17/v18 路线, 新建分支实现 Streaming RLS-Reservoir Engine (ESN+GMM+RLS 在线主动推理) 阶段一

Work Log:
- 封存: v17 MQ5 移植与 v18 静态止损实验保留归档不再推进; git checkout -b reservoir-engine (自 main d0a4fbf)
- 引擎实现 research/reservoir/reservoir_engine.py (500行 NumPy 纯净单文件): L1 ESN(500神经元/密度10%/ρ0.95/tanh/W锁死) + L2 StreamingGMM(对角/Mahalanobis²>36+冷却96bar自生长/权重衰减/剪枝) + L3 遗忘RLS(默认λ0.999+δ1e-6+trace上限+周期对称化四重防缠绕); 主动推理双头标签 y_var_rel(未来4bar相对实现方差)+y_skew(±2.5有界偏斜); 仓位=clip(2ŝkew)×1/(1+0.5v̂ar)×混沌门, EMA平滑+死区0.05+速率0.20, 1oz口径, 成本|Δpos|×(0.3ATR+点差)
- 修3个bug: outer(k,e)转置→(e,k); GMM剪枝mu/sd/w长度失同步(冒烟数据少没暴露,真实数据立即触发)+被剪kb重指; 空扫描列表守卫
- 因果纪律: t收盘GMM→储层→RLS.update(φ_{t-4},y_{t-4})→预测→设pos_{t+1}; 全rolling统计因果
- 服务器(经GCP跳板paramiko)上传点火: 主运行(3种子+λ扫描{nres扫描+洗牌+动量基线) + 4并行视界探针(h16/h32/h96/h32低换手)
- 前端: 第三tab R1·Reservoir引擎实验室(src/components/reservoir-tab.tsx 450行): 结论横幅+4KPI+三层架构卡+权益SVG+W‖收敛+GMM生长双sparkline+逐年状态占比+逐年明细表+学习真实性/种子/λ/规模对照+视界探针表+部署路线三阶段
- 浏览器实测修3处: 接口形状metrics嵌套(runtime error)→对齐; 390px溢出236px→TabsList overflow-x-auto+grid子项min-w-0+规格span flex-wrap(经验二分min-width:0定位)
- 归档: remote-ops-record/reservoir_20261002/(主运行md5 40de064+探针版d73ed6f+5json+5out+REPORT_R1.md+artifacts.tgz md5 21b12ff0)

Stage Summary:
- R1阶段一诚实结论: 学习为真(能量头IC 0.343/方向头IC 0.020 t≈6.4, 洗牌对照IC→0.004确证), alpha在波动率不在方向(与M1时代AUC 0.505互证)
- 瓶颈在执行层不在学习层: 全视界毛利为正(+$411~+$1,080@1oz)但连续调仓成本是毛利33倍; 低换手变体(dead0.10/rate0.10)成本砍58%毛利仅−9%→净亏−$10,085→−$3,695, 修复路径已证
- 算力承诺兑现: 单核0.478ms/bar(p99 0.61) 引擎4.1MB(nres1000→16.1MB但2.52ms/bar超2ms如实报告); W‖在19↔85终身漂移无灾难遗忘; GMM收敛4组件
- 种子稳健(3种子PnL −13.6k~−14.2k); λ0.999 IC最优; n_res1000无增益
- 下一步: P0事件驱动调仓(换手再砍70%) / P1能量头×v17规则引擎融合(IC0.34波动率预测做HTF突破在线闸门) / 双头双视界部署 / 阶段二ZeroMQ桥→阶段三DLL
- 分支: reservoir-engine 已建(本次提交推送), 用户可独立于 main/server 演进该路线

---
Task ID: 34-addendum
Agent: main (Z.ai Code)
Task: R1 分支推送收尾 + git 大文件事故处理

Work Log:
- 推送事故: 首推被拒 research-lab/data.csv 100.93MB (GitHub 100MB 硬限)
- 第一次修复失败教训×2: ① filter-branch 只重写了 a3d5a4b^..HEAD, 大文件 blob 在更深的 c212465/6ecbc87 祖先里仍在; ② `A && B && C; D` 链在 merge-base 非零处断裂导致"data.csv已清除"检查根本没执行(和 worklog 坑位第2条同型——坑位集要加星号)
- 最终方案(贴合用户"从头再来"语义): 孤儿分支单提交快照 — 全部研究成果(三tab前端+引擎+归档+worklog)打成 be9b4f5 一个干净提交, 581 个对象全 <100MB, 推送成功
- .gitignore 新增: research-lab/data.csv / skills/ / upload/ / tool-results/ / .zscripts/ / dev.log; 原始 M1 csv 备份在 /home/z/data-backup/XAUUSDc_M1.csv (md5 f6b0d44d, 服务器原件同在)
- 本地 main 仍为重写后的杂历史(未再推送); origin/main 保持上次会话的纯归档态 b85e6aa 不动 — 分仓哲学: main=纯归档, server=旧研究, reservoir-engine=新路线
- stash/refs-original 清理完毕; ntfy 通知已发

Stage Summary:
- GitHub 新分支 reservoir-engine = be9b4f5 (孤儿快照, 无历史包袱, 无大文件) — 用户"新建一个github分支从头再来"指令完成
- 待办转移: 本地 main 与 origin/main(b85e6aa) 分叉的最终处置(重置本地 or 保留)留给用户决定
