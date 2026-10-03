//+------------------------------------------------------------------+
//|                                          XAUUSD_v17_Ultimate.mq5 |
//|      v17 终极版 · 冠军核心 + 2条因果能量纪律 · MT5 独立复验版      |
//+------------------------------------------------------------------+
//  研究档案  : remote-ops-record/v17u_20261003/CHAMPION_v17u.md
//  血统      : v17 冠军 (159笔/+$2,980.2/笔均$18.74/逐年全正) 在
//              v18(44止损变体) 与 v19(24个R1闸门变体) 两轮攻击下全部
//              存活之后, 由 v19-H3 能量诊断提炼出的第 5、6 条规则。
//
//  研究成绩  : XAUUSD M30, 2022-01-02 ~ 2026-07-17, 0.3×ATR 悲观成本
//              109 笔 · 总 PnL +$3,389.4 · 笔均 $31.10 (t=3.65)
//              逐年全正 { 2022:+313.0 · 2023:+362.6 · 2024:+98.8
//                        2025:+1131.4 · 2026:+1483.6 }
//              胜率 65.1% · maxDD −$106.8(冠军−$259.6 的一半)
//              成本阶梯: base $3,553.8 / pess03 $3,389.4 / pess05 $3,243.5
//              全部逐年全正 (−8.7%)
//              对照冠军: 总量 +13.8% · 笔均 +66% · t 3.13→3.65 · maxDD 减半
//
//  策略 = 6 条规则（无 ML / 无拟合参数 / 全因果决策）:
//   1. 进场 : 55 根已完成 M30 bar 的高点通道线（海龟 S2 停损单）。
//             触及即成交；跳空开盘已越过线则按开盘市价成交（保守口径）。
//             仅做多（研究: 空头负贡献）。
//   2. 闸门 : ATR14（TR 的 14 根简单均值——研究口径, 非 Wilder）高于其
//             自身过去一季度（3024 根 M30）中位数才允许开仓。
//   3. 能量 : 低能入场 —— 决策时刻的因果 nowcast
//             NR = 最近4根M15对数收益平方和 / 同序列EMA96基线
//             必须 NR < 1.15 才允许挂停损单（能量压缩期的突破更真；
//             诊断: 低能五分位笔均$30~33 vs 高能$8~12, 断崖结构）。
//   4. 封锁 : 高能触发跳过 → 32 根 M30 (16小时) 再武装封锁。触发线被
//             触及但 NR≥1.15 时本轮突破被跳过, 并封锁后续 32 根 bar 的
//             一切入场（消灭同一能量事件内的"追单"次级突破——它们笔均
//             仅 $11.5, 是终极版对冠军唯一的稀释源）。封锁期内再次出现
//             高能触发则顺延。
//   5. 出场 : 持满 5 个交易日（240 根 M30），第 240 根开盘市价平仓。
//             无止损（v18: 44 个静态灾难止损变体全部负期望；时间出场
//             本身就是该结构的灾难止损, 尾部管理靠仓位规模）。
//   6. 纪律 : 串行单仓——持仓期内忽略新信号；出场当根不再进场；
//             停损单仅当根有效, 逐根重估通道线。
//
//  ============ MT5 策略测试器用法 ============
//   · 品种 XAUUSD · 周期 M30 · 日期 2022.01.01 ~ 2026.07.31
//   · 模式: 『基于真实报价的每笔报价』最佳; 最低用『1 分钟 OHLC』
//     (『仅开盘价』无法模拟停损单 bar 内触发, 不要用)
//   · 默认 0.01 手 = 1 盎司 → 测试器 $ 数字 ≈ 研究 $ 口径
//   · 成本映射: 研究扣 0.3×ATR/RT 悲观成本(中位 $1.8/笔), MT5 真实
//     点差(金通常 $0.2~0.4/RT) → 预期落在 base($3,554)~pess03($3,389)
//     之间; 逐年全正 + 笔均 ≥ $24 视为复验通过
//   · 2026-07-17 之后(如有数据) = 真正的样本外前向检验
//   · A/B 对照: InpEnergyFilter=false 即回到纯 v17 冠军(159笔/$2,980)
//
//  ============ 与研究引擎的已知差异(诚实披露) ============
//   · 数据源不同: MT5 经纪商行情 vs 研究用清洗后 M1 聚合 → 笔数允许
//     ±10% 级别漂移
//   · NR 的 EMA96 在本 EA 用最近 ~600 根 M15 滚动计算(权重残差<1e-4),
//     研究为全历史 EMA —— 差异可忽略
//   · 停损单以 Ask 触发(引擎用数据 high), 入场贵约 1 个点差——已被
//     研究的 0.3×ATR 悲观成本覆盖
//   · 封锁状态(再武装倒计时)不跨 EA 重启持久化——重启后重新武装
//   · 诚实条款: 终极版的两条能量规则是事后细化(诊断驱动, 非预注册),
//     参数平台宽广(θ0.9~1.25 × 封锁2~96根全正, 笔均$22~34)但存在
//     选择偏差; 独立数据复验(MT5真实tick)正是本文件的用途
//
#property copyright "RiverMind Research Lab · v17 ultimate port (model-free)"
#property version   "1.00"
#property description "XAUUSD M30 v17终极版(无ML): Donchian-55停损单进场+只做多,"
#property description "ATR14>季度中位闸门, M15低能nowcast(NR<1.15)入场过滤,"
#property description "高能触发跳过后32根M30封锁再武装, 持满5交易日开盘平仓, 串行单仓."
#property description "研究参照(0.3xATR成本): 109笔 +$3389 笔均$31.1 逐年全正 maxDD-$107."

#include <Trade/Trade.mqh>

//---- 输入参数（默认 = v17 终极版；0 = 按图表周期自动推导）
input int    InpDonchianN       = 55;       // Donchian 通道周期(根, 已完成bar)
input int    InpATRPeriod       = 14;       // ATR 周期(TR简单均值, 研究口径)
input int    InpGateMode        = 1;        // 波动闸门 0=无 1=ATR>季度中位(冠军)
input int    InpGateWindow      = 0;        // 闸门回看窗口(根; 0=自动: 63交易日)
input int    InpGateMinBars     = 200;      // 闸门最少样本(研究 min_periods)
input int    InpHoldBars        = 0;        // 时间出场持仓(根; 0=自动: 5交易日)
input bool   InpEnergyFilter    = true;     // 能量纪律(终极版=开; 关=纯v17冠军)
input double InpNRTheta         = 1.15;     // 低能阈值θ(NR<θ才允许进场)
input int    InpNRBlockBars     = 32;       // 高能触发跳过后封锁根数(M30)
input double InpVolume          = 0.01;     // 每笔手数(0.01lot=1oz → $≈研究口径)
input bool   InpAllowShort      = false;    // 允许空头(冠军=只做多)
input bool   InpUseDisasterStop = false;    // 静态灾难止损(v18: 44变体全负期望)
input double InpDisasterATR     = 4.0;      // 灾难止损宽度(×入场决策时ATR)
input long   InpMagic           = 20261003; // 魔术号
input string InpComment         = "v17-ult";

//---- 能量 nowcast 常量(研究口径, 非输入)
#define NR_VP_WIN    4          // v4p = 4根M15对数收益平方和
#define NR_EMA_SPAN  96         // 基线EMA span
#define NR_M15_NEED  620        // M15 取数量(EMA96预热~300根, 620充裕)
#define NR_CLIP      6.0        // NR 上限(研究 clip)

//---- 全局状态
CTrade   trade;
datetime g_lastBar      = 0;     // 最新处理过的bar开盘时刻
datetime g_entryBarTime = 0;     // 入场bar开盘时刻
double   g_entryPrice   = 0.0;   // 入场成交价(记账)
datetime g_blockUntil   = 0;     // 再武装封锁截止时刻(高能触发跳过)
double   g_skipLine     = 0.0;   // 上一根bar因高能跳过时的通道线(触线检查)
datetime g_skipBarTime  = 0;     // 上一根被跳过bar的开盘时刻
int      g_donN         = 55;
int      g_gateWin      = 3024;
int      g_hold         = 240;
int      g_digits       = 2;

//---- 统计(测试器日志汇总)
int      g_nIn = 0, g_nOut = 0, g_nBlocked = 0, g_nSkipHighE = 0;
double   g_totalPnl = 0.0;
#define  YEAR_BASE 2022
#define  YEAR_SPAN 16
double   g_yearPnl[YEAR_SPAN];
int      g_yearN[YEAR_SPAN];

//+------------------------------------------------------------------+
//| 初始化                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber((ulong)InpMagic);
   trade.SetDeviationInPoints(100);
   trade.SetTypeFillingBySymbol(_Symbol);

   int    ps  = PeriodSeconds(PERIOD_CURRENT);
   int    bpd = 86400 / ps;
   g_donN    = (InpDonchianN > 0)  ? InpDonchianN  : 55;
   g_gateWin = (InpGateWindow > 0) ? InpGateWindow : 63 * bpd;
   g_hold    = (InpHoldBars   > 0) ? InpHoldBars   : 5  * bpd;
   g_digits  = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   ArrayInitialize(g_yearPnl, 0.0);
   ArrayInitialize(g_yearN, 0);

   ulong tk = FindMyPosition();
   if(tk > 0)
   {
      datetime pt = (datetime)PositionGetInteger(POSITION_TIME);
      g_entryBarTime = iTime(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_CURRENT, pt));
      g_entryPrice   = PositionGetDouble(POSITION_PRICE_OPEN);
      PrintFormat("[v17u] 恢复已有仓位: 入场bar=%s @%.2f",
                  TimeToString(g_entryBarTime, TIME_DATE | TIME_MINUTES), g_entryPrice);
   }

   Print("======================================== v17 终极版 · MT5 复验 =======");
   PrintFormat("[v17u] 品种=%s 周期=%s | Donchian=%d | ATR=%d(SMA of TR) | 闸门=%s(窗=%d) | 持仓=%d根 | 手数=%.2f",
               _Symbol, EnumToString(_Period), g_donN, InpATRPeriod,
               InpGateMode == 0 ? "无" : "ATR>季中位", g_gateWin, g_hold, InpVolume);
   PrintFormat("[v17u] 能量纪律=%s | θ=%.2f | 封锁=%d根M30 | NR=4根M15收益²和/EMA%d(纯因果)",
               InpEnergyFilter ? "ON(终极版)" : "OFF(纯v17冠军)", InpNRTheta, InpNRBlockBars, NR_EMA_SPAN);
   PrintFormat("[v17u] 研究参照(0.3xATR成本): 终极版 109笔 +$3389 笔均$31.1 逐年全正{313/363/99/1131/1484} 胜率65.1%% maxDD-$107");
   PrintFormat("[v17u] 研究参照(0.3xATR成本): 冠军版 159笔 +$2980 笔均$18.7 逐年全正{243/165/320/1395/859} 胜率58.5%% maxDD-$260");
   Print("[v17u] 建议: 品种XAUUSD 周期M30 日期2022.01.01-2026.07.31 模式=每笔报价(真实报价)或1分钟OHLC");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| 收尾: 打印年度汇总供与研究对照                                     |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("======================================== v17 终极版 · MT5 回测汇总 =======");
   PrintFormat("[v17u] 成交=%d 平仓=%d 总PnL≈$%.2f (0.01lot=1oz口径, 含真实点差) | 高能跳过=%d次 封锁生效=%d次",
               g_nIn, g_nOut, g_totalPnl, g_nSkipHighE, g_nBlocked);
   for(int i = 0; i < YEAR_SPAN; i++)
      if(g_yearN[i] > 0)
         PrintFormat("[v17u]   %d年: %3d笔  $%+.2f", YEAR_BASE + i, g_yearN[i], g_yearPnl[i]);
   Print("[v17u] 判定: 逐年全正 ∧ 笔均≥$24 为复验通过; 笔均预期 $29~33(pess03~base)");
   Print("[v17u] 2026-07-17之后的数据 = 样本外前向检验(研究未考核)");
}

//+------------------------------------------------------------------+
//| 主循环: 只在新bar开盘做决策(bar内触发交给停损单)                    |
//+------------------------------------------------------------------+
void OnTick()
{
   if(g_entryBarTime == 0)
   {
      ulong tk = FindMyPosition();
      if(tk > 0)
      {
         datetime pt = (datetime)PositionGetInteger(POSITION_TIME);
         g_entryBarTime = iTime(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_CURRENT, pt));
         g_entryPrice   = PositionGetDouble(POSITION_PRICE_OPEN);
      }
   }

   // ---- 新bar检测 ----
   datetime cur = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(cur == 0 || cur == g_lastBar)
      return;
   bool firstBar = (g_lastBar == 0);
   g_lastBar = cur;

   // (0) 清掉上一根bar遗留的挂单(停损单仅当根有效)
   DeleteStaleOrders();

   // (0.5) 高能跳过的触线检查: 上一根bar(因NR≥θ未挂单)的high是否触及当时通道线
   //       触及 → 该突破被判跳过 → 封锁 InpNRBlockBars 根(引擎: blocked_until=k+block)
   if(!firstBar && g_skipLine > 0.0)
   {
      double prevHigh = iHigh(_Symbol, PERIOD_CURRENT, 1);
      if(prevHigh >= g_skipLine)
      {
         datetime until = g_skipBarTime + (datetime)(InpNRBlockBars * PeriodSeconds(PERIOD_CURRENT));
         if(until > g_blockUntil)
            g_blockUntil = until;
         g_nSkipHighE++;
         PrintFormat("[v17u] 高能触发被跳过: bar=%s NR≥%.2f 触线%.2f → 封锁至 %s",
                     TimeToString(g_skipBarTime, TIME_DATE | TIME_MINUTES), InpNRTheta, g_skipLine,
                     TimeToString(g_blockUntil, TIME_DATE | TIME_MINUTES));
      }
      g_skipLine = 0.0;
   }

   // (1) 时间出场: 持满 g_hold 根 → 本根开盘市价平仓
   bool exitedThisBar = false;
   ulong posTicket = FindMyPosition();
   if(posTicket > 0 && g_entryBarTime > 0)
   {
      int barsHeld = iBarShift(_Symbol, PERIOD_CURRENT, g_entryBarTime, false);
      if(barsHeld >= g_hold)
      {
         if(trade.PositionClose(posTicket))
         {
            exitedThisBar = true;
            g_entryBarTime = 0;
            PrintFormat("[v17u] TIME-EXIT: 持满%d根, 开盘平仓(%.2f)", g_hold,
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));
         }
      }
   }
   if(posTicket > 0)   return;    // 串行单仓: 持仓期忽略新信号
   if(exitedThisBar)   return;    // 引擎语义: 出场当根不再进场(也不更新封锁)

   // (2) 空仓: 评估本根进场
   TryEnter(cur);
}

//+------------------------------------------------------------------+
//| 进场评估: 闸门开 ∧ (能量纪律关 ∨ NR<θ且未封锁) → 挂停损单          |
//+------------------------------------------------------------------+
void TryEnter(const datetime curBarTime)
{
   double levelUp = 0.0, levelDn = 0.0, atrPrev = 0.0;
   bool   gateOn  = false;
   if(!ComputeSignals(levelUp, levelDn, atrPrev, gateOn))
      return;
   if(!gateOn || atrPrev <= 0.0)
      return;                       // 闸门关(低波期禁开仓)

   // ---- 能量纪律检查(终极版核心) ----
   bool   blocked = false;
   if(InpEnergyFilter)
   {
      double nr = NowcastNR();
      if(!(nr < InpNRTheta))        // NR≥θ 或 M15数据不足(NaN→高能, 保守)
      {
         // 高能: 本根不挂单; 记录通道线供下一根开盘做触线→封锁判定
         g_skipLine    = levelUp;
         g_skipBarTime = curBarTime;
         return;
      }
      blocked = (curBarTime <= g_blockUntil);
      if(blocked)
      {
         g_nBlocked++;
         return;                    // 封锁期内低能触发: 不进场也不顺延(引擎语义)
      }
   }

   double point   = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   datetime expiry = curBarTime + PeriodSeconds(PERIOD_CURRENT);
   double vol      = NormalizeVolume(InpVolume);

   // ---- 多头(冠军方向) ----
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0) return;
   if(ask >= levelUp)
   {
      // 跳空/本根开盘已越过通道线 → 按开盘市价成交(保守口径, 同引擎)
      double sl = InpUseDisasterStop ? NormalizeDouble(ask - InpDisasterATR * atrPrev, g_digits) : 0.0;
      if(!trade.Buy(vol, _Symbol, 0.0, sl, 0.0, InpComment))
         LogTradeError("Buy(gap)");
   }
   else
   {
      double sl = InpUseDisasterStop ? NormalizeDouble(levelUp - InpDisasterATR * atrPrev, g_digits) : 0.0;
      double px = NormalizeDouble(levelUp, g_digits);
      if(levelUp - ask <= minDist)
      {
         if(!trade.Buy(vol, _Symbol, 0.0, sl, 0.0, InpComment))
            LogTradeError("Buy(neartouch)");
      }
      else if(!trade.BuyStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_SPECIFIED, expiry, InpComment))
      {
         if(!trade.BuyStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_GTC, 0, InpComment))
            LogTradeError("BuyStop");
      }
   }

   // ---- 空头(默认关; 研究: 空头负贡献; 能量纪律仅校验多头方向) ----
   if(!InpAllowShort) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(bid <= 0.0) return;
   if(bid <= levelDn)
   {
      double sl = InpUseDisasterStop ? NormalizeDouble(bid + InpDisasterATR * atrPrev, g_digits) : 0.0;
      if(!trade.Sell(vol, _Symbol, 0.0, sl, 0.0, InpComment))
         LogTradeError("Sell(gap)");
   }
   else
   {
      double sl = InpUseDisasterStop ? NormalizeDouble(levelDn + InpDisasterATR * atrPrev, g_digits) : 0.0;
      double px = NormalizeDouble(levelDn, g_digits);
      if(bid - levelDn <= minDist)
      {
         if(!trade.Sell(vol, _Symbol, 0.0, sl, 0.0, InpComment))
            LogTradeError("Sell(neartouch)");
      }
      else if(!trade.SellStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_SPECIFIED, expiry, InpComment))
      {
         if(!trade.SellStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_GTC, 0, InpComment))
            LogTradeError("SellStop");
      }
   }
}

//+------------------------------------------------------------------+
//| 因果能量 nowcast: NR = v4p(last完成M15) / EMA96(v4p)               |
//| v4p(t) = 最近4根M15对数收益平方和(含t) — 与研究 nowcast_series 同款 |
//| 决策bar = 最后先于本M30 bar开盘收盘的M15 bar (=rates[1])            |
//| 返回 -1 = M15历史不足(视为高能, 保守不进场)                         |
//+------------------------------------------------------------------+
double NowcastNR()
{
   MqlRates r[];
   ArraySetAsSeries(r, true);
   int got = CopyRates(_Symbol, PERIOD_M15, 0, NR_M15_NEED, r);
   if(got < NR_EMA_SPAN + NR_VP_WIN + 20)
      return -1.0;

   // ret[j] = log(close[j]/close[j+1]), j=1..got-2 (series: 0=形成中, 大索引=更老)
   int nRet = got - 2;                     // 可用收益根数(j=1..nRet)
   double v4p[];
   int nV = nRet - NR_VP_WIN + 1;          // v4p 索引 j=1..nV (j 的4根窗口=j..j+3)
   if(nV < NR_EMA_SPAN + 10)
      return -1.0;
   ArrayResize(v4p, nV + 1);
   for(int j = 1; j <= nV; j++)
   {
      double s = 0.0;
      for(int m = 0; m < NR_VP_WIN; m++)
      {
         double rc = MathLog(r[j + m].close / r[j + m + 1].close);
         s += rc * rc;
      }
      v4p[j] = s;
   }
   // EMA96 over v4p 序列(从老到新: j=nV → j=1), adjust=False, 初值0(同pandas前导0语义)
   double alpha = 2.0 / (NR_EMA_SPAN + 1.0);
   double ema   = 0.0;
   for(int j = nV; j >= 1; j--)
      ema += alpha * (v4p[j] - ema);
   if(ema <= 1e-14)
      return -1.0;
   double nr = v4p[1] / ema;
   if(nr > NR_CLIP) nr = NR_CLIP;
   return nr;                              // 决策M15 bar = rates[1] = [T-15m, T)
}

//+------------------------------------------------------------------+
//| 信号计算: Donchian通道 + ATR(SMA of TR) + 波动闸门                 |
//| 全部只用已完成bar(序列索引1..N) —— 无回望                          |
//+------------------------------------------------------------------+
bool ComputeSignals(double &levelUp, double &levelDn, double &atrPrev, bool &gateOn)
{
   levelUp = 0.0; levelDn = 0.0; atrPrev = 0.0; gateOn = false;

   int need = MathMax(g_gateWin, g_donN) + InpATRPeriod + 4;
   double h[], l[], c[];
   ArraySetAsSeries(h, true);
   ArraySetAsSeries(l, true);
   ArraySetAsSeries(c, true);
   int gh = CopyHigh(_Symbol, PERIOD_CURRENT, 0, need, h);
   int gl = CopyLow(_Symbol, PERIOD_CURRENT, 0, need, l);
   int gc = CopyClose(_Symbol, PERIOD_CURRENT, 0, need, c);
   int got = MathMin(gh, MathMin(gl, gc));
   if(got < g_donN + 2)
      return false;

   double hi = h[1], lo = l[1];
   for(int i = 2; i <= g_donN; i++)
   {
      if(h[i] > hi) hi = h[i];
      if(l[i] < lo) lo = l[i];
   }
   levelUp = hi;
   levelDn = lo;

   int trMax = got - 2;
   if(trMax < 1 + InpATRPeriod)
      return true;
   double S[];
   ArrayResize(S, trMax + 1);
   S[0] = 0.0;
   for(int i = 1; i <= trMax; i++)
   {
      double tr = MathMax(h[i] - l[i],
                  MathMax(MathAbs(h[i] - c[i + 1]), MathAbs(l[i] - c[i + 1])));
      S[i] = S[i - 1] + tr;
   }
   int ap = InpATRPeriod;
   atrPrev = (S[ap] - S[0]) / ap;
   if(atrPrev <= 0.0)
      return true;

   if(InpGateMode == 0)
   {
      gateOn = true;
      return true;
   }

   int maxJ = MathMin(g_gateWin, trMax - ap + 1);
   if(maxJ < InpGateMinBars)
      return true;
   double a[];
   ArrayResize(a, maxJ);
   for(int j = 1; j <= maxJ; j++)
      a[j - 1] = (S[j + ap - 1] - S[j - 1]) / ap;
   double q   = (InpGateMode == 2) ? 0.30 : 0.50;
   double ref = QuantileSorted(a, q);
   gateOn = (atrPrev > ref);
   return true;
}

//+------------------------------------------------------------------+
//| 分位数(升序排序后线性插值, 同 pandas 默认)                          |
//+------------------------------------------------------------------+
double QuantileSorted(double &a[], double q)
{
   int n = ArraySize(a);
   if(n == 0) return EMPTY_VALUE;
   ArraySort(a);
   double pos = q * (n - 1);
   int lo = (int)MathFloor(pos);
   int hi = (int)MathCeil(pos);
   if(lo == hi) return a[lo];
   return a[lo] + (pos - lo) * (a[hi] - a[lo]);
}

//+------------------------------------------------------------------+
//| 工具: 找本策略仓位 / 清挂单 / 手数规格化 / 错误日志                 |
//+------------------------------------------------------------------+
ulong FindMyPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic)  continue;
      return tk;
   }
   return 0;
}

void DeleteStaleOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong tk = OrderGetTicket(i);
      if(tk == 0) continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;
      if(OrderGetInteger(ORDER_MAGIC) != InpMagic)  continue;
      if(!trade.OrderDelete(tk))
         LogTradeError("OrderDelete");
   }
}

double NormalizeVolume(double v)
{
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step > 0.0) v = MathRound(v / step) * step;
   return MathMin(MathMax(v, vmin), vmax);
}

void LogTradeError(string what)
{
   PrintFormat("[v17u] %s 失败: retcode=%u (%s)",
               what, trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| 成交事件: 记录入场bar + 出场PnL年度记账                            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(trans.symbol != _Symbol)                  return;
   if(!HistoryDealSelect(trans.deal))           return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic) return;

   long     entryType = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   double   price     = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
   datetime dtm       = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
   double   vol       = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   long     dtype     = HistoryDealGetInteger(trans.deal, DEAL_TYPE);

   if(entryType == DEAL_ENTRY_IN)
   {
      g_entryBarTime = iTime(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_CURRENT, dtm));
      g_entryPrice   = price;
      g_nIn++;
      double nr = InpEnergyFilter ? NowcastNR() : 0.0;
      PrintFormat("[v17u] ENTRY #%d %s @ %.2f vol=%.2f NR=%.2f bar=%s → 持满%d根后开盘平仓",
                  g_nIn, (dtype == DEAL_TYPE_BUY ? "LONG" : "SHORT"),
                  price, vol, nr,
                  TimeToString(g_entryBarTime, TIME_DATE | TIME_MINUTES), g_hold);
   }
   else if(entryType == DEAL_ENTRY_OUT || entryType == DEAL_ENTRY_OUT_BY)
   {
      double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      double pnl;
      if(dtype == DEAL_TYPE_SELL) pnl = (price - g_entryPrice) * vol * contract;
      else                        pnl = (g_entryPrice - price) * vol * contract;
      long   reason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
      string rsn    = (reason == DEAL_REASON_SL) ? "灾难止损" : "时间出场/强平";

      g_nOut++;
      g_totalPnl += pnl;
      MqlDateTime dt;
      TimeToStruct(dtm, dt);
      int yi = dt.year - YEAR_BASE;
      if(yi >= 0 && yi < YEAR_SPAN) { g_yearPnl[yi] += pnl; g_yearN[yi]++; }

      PrintFormat("[v17u] EXIT  #%d %s @ %.2f  pnl≈$%+.2f (%s)",
                  g_nOut, (dtype == DEAL_TYPE_SELL ? "close-L" : "close-S"),
                  price, pnl, rsn);
      if(g_entryBarTime != 0) g_entryBarTime = 0;
   }
}
//+------------------------------------------------------------------+
