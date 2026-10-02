# XAUUSD 量化基础设施 恢复作战档案

> 沙箱多次重置后重建。凭据、通道、坑位、当前状态尽在此处。

## 凭据表
| 资产 | 值 |
|---|---|
| GCP 跳板 | root@35.208.137.216:22 密码 `Abc147258@`（Debian 12, OpenSSH 9.2）|
| **当前目标机** | sx01-ssh.gpuhome.cc:**30133** root 密码 `ryq94rge`（30核/80G cgroup 实锤，jupyter-g0ue1ecf2evh9rr9，49G 持久卷 /root/rivermind-data）|
| ~~旧目标机~~ | ~~sx01-ssh.gpuhome.cc:30181 密码 03g2rfg3~~ **已释放，连接拒绝** |
| GitHub | hjw221/a（server 分支 head db08ce0）；私钥沙箱重置后**丢失**（只影响 push，clone/archive 拉取不受影响）|
| ntfy 通道 | CMD=`xauusd-qv7m2zk9-cmd`（发指令）/ RES=`xauusd-qv7m2zk9-res`（收汇报）|

## 工具三件套（recovery-tools/）
- `jump-ssh.py` — 经跳板执行命令（TGT_PORT/TGT_PASS 环境变量切机器）
- `stream-upload.py` — exec 通道流式上传（~5-22KB/s，SFTP 卡死时的替代）
- `relay-upload.py` — 三段式中继（沙箱→跳板→原生ssh，本期已证无效：路径级限速）
- `jump_ssh_lib.py` — 连接库

## 2026-09-30 新机部署纪要（Task 15）
1. **新机网络格局（全量实测）**：
   - 跳板→新机 任何方式（SFTP/exec/原生ssh/并行）：~3-30KB/s **路径级限速**（并行4路聚合不变）
   - 新机→GitHub codeload：33KB/s 稳定可用的**唯一国际通道**
   - 新机→raw.githubusercontent / ghfast.top / gh-proxy.com / ghproxy.net：全挂
   - 新机→国内（清华 pip）：**快**（几百MB包4分钟）→ 依赖安装走 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`
   - ntfy.sh 可达但免费版附件限额小（8MB 被拒）
   - 新机→跳板反拉（sftp://GCP:22）：10KB/s 同样被限
2. **部署方案（已执行）**：
   - 代码包（428KB）走 stream-upload exec 流式 88s 到位，md5 ✓，py_compile 全过
   - `/root/rivermind-fs` → `/root/rivermind-data` 软链（所有脚本硬编码 rivermind-fs 一招全解）
   - 数据包（26.7MB tarball）走 **服务器侧 nohup 自主脚本** `/tmp/auto_deploy.sh`：codeload 33KB/s 循环下载（40次重试）→ tar/gz/csv 三重 md5 → gunzip → phase3 每臂 ML_THREADS=7 补丁 → ML_THREADS=26 点火 run_master.sh，全程 ntfy 汇报
3. **线程布局**：phase1/2 单进程 26 线程；phase3 四臂并行各 7 线程（4×7=28/30核）
4. **agent.py** pid 10230 已上线（ntfy 双向通道，手机可发 status/ps/gpu/tail 等白名单指令）
5. **全跑一遍**（用户指示）：不用 1 核机 checkpoint 断点（temp_server_run_20250929 留在仓库做对照），单环境全新跑

## 管线结构（server 分支 db08ce0）
- BASE=/root/rivermind-fs/xauusd（软链→rivermind-data）
- Phase1 v3bal_ens 重跑 → Phase2 m1sc_ad prep+tune → Phase3 四臂（A0 m1sc_ad / A1 ens / A2 ens6 / A3 ens_hl180）
- run_m1.py L322 支持断点续跑（checkpoint 存在自动跳折）
- 历史基线对照：m1sc_ad 6496笔/35.2%/PLR 2.05/+$732.7/Sharpe 2.29；v3bal_ens 1611笔/PLR 2.57/+$1113.5/Sharpe 1.74

## 坑位速查（血泪）
1. `pkill -f "关键字"` 会匹配自己所在会话的命令行 → 自杀。用 PID 定点或 `grep "[k]eyword"` 写法
2. `cmd | tail -1` 退出码是 tail 的（bootstrap v1.1 死代码根因）
3. 容器 nproc/free 显示宿主机规格，**cgroup cpu.max/memory.max 才是真相**（1核机教训；30181 已验 30核/80G 为真）
4. gpuhome 每个容器端口 QoS 独立：30181 曾有 1.4MB/s 跳板通道，30133 只有 ~30KB/s
5. bash heredoc 里嵌 `$(...)` 会被本地先展开——远程脚本用 sftp.open 写文件最稳
6. codeload.github.com 不支持 Range 续传（rc=33），失败只能整包重来
