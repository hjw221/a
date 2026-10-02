//+------------------------------------------------------------------+
//|                                          XAUUSD_v17_Champion.mq5 |
//|          v17 HTF 突破冠军 · MT5 独立复验版（纯规则 · 零模型）      |
//+------------------------------------------------------------------+
//  研究档案  : remote-ops-record/v17_20261002/CHAMPION_v17.md
//  研究成绩  : XAUUSD M30, 2022-01-02 ~ 2026-07-17, 0.3×ATR 悲观成本口径
//              159 笔 · 总 PnL +$2,980.2 · 笔均 $18.74 (t=3.13)
//              逐年全正 { 2022:+242.5/34笔 · 2023:+164.5/38笔
//                        2024:+319.8/36笔 · 2025:+1394.5/36笔
//                        2026H1:+858.9/15笔 }
//              胜率 58.5% · PLR 2.19 · Sharpe 1.36 · maxDD −$259.6
//
//  策略 = 4 条规则（无任何 ML / 模型 / 拟合参数）:
//   1. 进场 : 55 根已完成 M30 bar 的高点通道线（海龟 S2 停损单）。
//             价格触及通道线上方即停损单成交；跳空开盘已越过线则按开盘市价成交
//             （保守口径，与研究引擎一致）。仅做多（研究中空头负贡献）。
//   2. 闸门 : ATR14（TR 的 14 根简单均值——研究口径，非 Wilder）高于其自身
//             过去一季度（3024 根 M30）中位数才允许开仓。决策只用已完成 bar。
//   3. 出场 : 持满 5 个交易日（240 根 M30），第 240 根开盘市价平仓。无止损。
//             （v18 分支A 已检验 44 个静态灾难止损变体全部负期望 → 默认关闭，
//               时间出场本身就是该结构的灾难止损，尾部管理靠仓位规模。）
//   4. 纪律 : 串行单仓——持仓期内忽略新信号；出场当根不再进场（引擎语义）；
//             停损单仅当根有效，逐根重估通道线（与引擎逐 bar 重算一致）。
//
//  ============ MT5 策略测试器用法 ============
//   · 品种 XAUUSD · 周期 M30 · 日期 2022.01.01 ~ 2026.07.31（研究窗）
//   · 模式：『基于真实报价的每笔报价』最佳；最低用『1 分钟 OHLC』
//     （『仅开盘价』模式无法正确模拟停损单的 bar 内触发，不要用）
//   · 默认 0.01 手 = 1 盎司 → 测试器里的 $ 数字 ≈ 研究报告的 $ 口径
//   · 成本映射：研究扣 0.3×ATR/RT 悲观成本（中位 $1.37/笔），MT5 用真实
//     点差（金通常 $0.2~0.4/RT）→ 预期结果落在研究 base($3,208)~pess03($2,980) 之间
//   · 2026-07-17 之后（如有数据）= 真正的样本外前向检验
//
//  ============ 与研究引擎的已知差异（诚实披露） ============
//   · 数据源不同：MT5 经纪商行情 vs 研究用清洗后 M1 聚合（剔除了 tickvol≤5
//     且波幅<$0.01 的死 bar 与 15σ 异常 bar）→ 笔数允许 ±10% 级别漂移
//   · 服务器时区不同会平移日内 session，但本策略全部是 bar 计数结构
//     （55 根通道 / 240 根持仓 / 3024 根闸门），对时区不敏感
//   · 停损单以 Ask 触发（引擎用数据 high），入场贵约 1 个点差——已被研究
//     的 0.3×ATR 悲观成本假设覆盖
//   · 双向同根触发保守跳过（引擎行为）仅在开启 InpAllowShort 时相关，默认关
//
#property copyright "RiverMind Research Lab · v17 champion port (model-free)"
#property version   "1.00"
#property description "XAUUSD M30 纯规则突破冠军(无ML): Donchian-55 停损单进场, 只做多,"
#property description "ATR14>季度中位波动闸门, 持满5个交易日(240根M30)开盘平仓, 串行单仓."
#property description "研究参照(0.3xATR成本): 159笔 +$2980, 2022-2026 逐年全正."

#include <Trade/Trade.mqh>

//---- 输入参数（默认 = v17 冠军；0 = 按图表周期自动推导）
input int    InpDonchianN       = 55;       // Donchian 通道周期(根, 已完成bar)
input int    InpATRPeriod       = 14;       // ATR 周期(TR简单均值, 研究口径, 非Wilder)
input int    InpGateMode        = 1;        // 波动闸门 0=无 1=ATR>季度中位(冠军) 2=ATR>p30
input int    InpGateWindow      = 0;        // 闸门回看窗口(根; 0=自动: 63交易日)
input int    InpGateMinBars     = 200;      // 闸门最少样本(研究 min_periods)
input int    InpHoldBars        = 0;        // 时间出场持仓(根; 0=自动: 5交易日)
input double InpVolume          = 0.01;     // 每笔手数(0.01lot=1oz → $≈研究口径)
input bool   InpAllowShort      = false;    // 允许空头(冠军=只做多; 研究空头负贡献)
input bool   InpUseDisasterStop = false;    // 静态灾难止损(v18: 44变体全负期望, 默认关)
input double InpDisasterATR     = 4.0;      // 灾难止损宽度(×入场决策时ATR)
input long   InpMagic           = 20261002; // 魔术号
input string InpComment         = "v17-champ";

//---- 全局状态
CTrade   trade;
datetime g_lastBar      = 0;     // 最新处理过的bar开盘时刻(新bar检测)
datetime g_entryBarTime = 0;     // 入场bar开盘时刻(逐bar计数用)
double   g_entryPrice   = 0.0;   // 入场成交价(记账用)
int      g_donN         = 55;    // 生效参数
int      g_gateWin      = 3024;
int      g_hold         = 240;
int      g_digits       = 2;

//---- 统计(测试器日志汇总)
int      g_nIn = 0, g_nOut = 0;
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
   int    bpd = 86400 / ps;                       // 每交易日bar数(M30=48/M15=96/H1=24, 同研究)
   g_donN    = (InpDonchianN > 0)              ? InpDonchianN              : 55;
   g_gateWin = (InpGateWindow > 0)             ? InpGateWindow             : 63 * bpd;
   g_hold    = (InpHoldBars    > 0)            ? InpHoldBars               : 5  * bpd;
   g_digits  = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   ArrayInitialize(g_yearPnl, 0.0);
   ArrayInitialize(g_yearN, 0);

   // 重启/换参恢复: 已有本策略仓位 → 从 POSITION_TIME 找回入场bar
   ulong tk = FindMyPosition();
   if(tk > 0)
   {
      datetime pt = (datetime)PositionGetInteger(POSITION_TIME);
      g_entryBarTime = iTime(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_CURRENT, pt));
      g_entryPrice   = PositionGetDouble(POSITION_PRICE_OPEN);
      PrintFormat("[v17] 恢复已有仓位: 入场bar=%s @%.2f",
                  TimeToString(g_entryBarTime, TIME_DATE | TIME_MINUTES), g_entryPrice);
   }

   Print("======================================== v17 冠军 · MT5 复验 =======");
   PrintFormat("[v17] 品种=%s 周期=%s | Donchian=%d | ATR=%d(SMA of TR) | 闸门=%s(窗=%d根, min=%d) | 持仓=%d根 | 手数=%.2f%s",
               _Symbol, EnumToString(_Period), g_donN, InpATRPeriod,
               InpGateMode == 0 ? "无" : (InpGateMode == 1 ? "ATR>季中位" : "ATR>p30"),
               g_gateWin, InpGateMinBars, g_hold, InpVolume,
               InpUseDisasterStop ? StringFormat(" | 灾难止损=%.1f×ATR(注意:v18结论为负期望)", InpDisasterATR) : " | 无止损(冠军口径)");
   PrintFormat("[v17] 研究参照(0.3×ATR成本): 159笔 +$2980 逐年{2022:+242.5/34, 2023:+164.5/38, 2024:+319.8/36, 2025:+1394.5/36, 2026:+858.9/15} 胜率58.5%%");
   Print("[v17] 建议: 品种XAUUSD 周期M30 日期2022.01.01-2026.07.31 模式=每笔报价(真实报价)或1分钟OHLC");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| 收尾: 打印年度汇总供与研究对照                                     |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("======================================== v17 MT5 回测汇总 =======");
   PrintFormat("[v17] 成交笔数=%d  平仓笔数=%d  总PnL≈$%.2f (0.01lot=1oz口径, 含真实点差)",
               g_nIn, g_nOut, g_totalPnl);
   for(int i = 0; i < YEAR_SPAN; i++)
      if(g_yearN[i] > 0)
         PrintFormat("[v17]   %d年: %3d笔  $%+.2f", YEAR_BASE + i, g_yearN[i], g_yearPnl[i]);
   Print("[v17] 研究参照: 逐年全正才是通过; 单笔均额预期 $18~20(pess03~base之间)");
   Print("[v17] 2026-07-17之后的数据 = 样本外前向检验(研究未考核)");
}

//+------------------------------------------------------------------+
//| 主循环: 只在新bar开盘做决策(bar内触发交给停损单/SL)                 |
//+------------------------------------------------------------------+
void OnTick()
{
   // 安全网: 仓位存在但未记录入场bar(OnTradeTransaction 之外的兜底)
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
   g_lastBar = cur;

   // (0) 清掉上一根bar遗留的挂单(停损单仅当根有效, 引擎逐根重估通道线)
   DeleteStaleOrders();

   // (1) 时间出场: 持满 g_hold 根 → 本根(第g_hold根)开盘市价平仓 = 引擎 o[k]
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
            g_entryBarTime = 0;   // PnL 记账在 OnTradeTransaction 的 OUT deal 里
            PrintFormat("[v17] TIME-EXIT: 持满%d根, 开盘平仓(%.2f)", g_hold,
                        SymbolInfoDouble(_Symbol, SYMBOL_BID));
         }
      }
   }
   if(posTicket > 0)   return;    // 串行单仓: 持仓期忽略新信号
   if(exitedThisBar)   return;    // 引擎语义: 出场当根不再进场

   // (2) 空仓: 评估本根进场(闸门 + 通道线 → 挂停损单/跳空市价)
   TryEnter(cur);
}

//+------------------------------------------------------------------+
//| 进场评估: 闸门开 + 通道线有效 → 挂本根有效的停损单                  |
//+------------------------------------------------------------------+
void TryEnter(const datetime curBarTime)
{
   double levelUp = 0.0, levelDn = 0.0, atrPrev = 0.0;
   bool   gateOn  = false;
   if(!ComputeSignals(levelUp, levelDn, atrPrev, gateOn))
      return;                       // 历史不足连通道都算不出
   if(!gateOn || atrPrev <= 0.0)
      return;                       // 闸门关(低波期禁开仓, 避2022型绞肉)

   double point   = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   datetime expiry = curBarTime + PeriodSeconds(PERIOD_CURRENT);   // 停损单仅本根有效
   double vol      = NormalizeVolume(InpVolume);

   // ---- 多头(冠军方向) ----
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0) return;
   if(ask >= levelUp)
   {
      // 跳空/本根开盘已越过通道线 → 按开盘市价成交(保守口径, 同引擎 entry=o[k])
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
         // 距离小于 stops level, 挂单会被拒 → 视为已触发, 市价进场
         if(!trade.Buy(vol, _Symbol, 0.0, sl, 0.0, InpComment))
            LogTradeError("Buy(neartouch)");
      }
      else if(!trade.BuyStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_SPECIFIED, expiry, InpComment))
      {
         // 经纪商可能不支持 Specified 过期 → 退 GTC, 下一根开盘手动清理兜底
         if(!trade.BuyStop(vol, px, _Symbol, sl, 0.0, ORDER_TIME_GTC, 0, InpComment))
            LogTradeError("BuyStop");
      }
   }

   // ---- 空头(默认关; 研究: 空头几乎全线负贡献) ----
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
   // 注: 引擎对"同根双方向都触发"保守跳过; MT5两侧挂单均可能成交(仅AllowShort时相关)
}

//+------------------------------------------------------------------+
//| 信号计算: Donchian通道 + ATR(SMA of TR) + 波动闸门                 |
//| 全部只用已完成bar(序列索引1..N, 0=形成中bar) —— 无回望              |
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
      return false;                          // 连 Donchian 都算不出

   // Donchian 通道线: 已完成bar 1..N 的最高/最低(=引擎 dhiN.shift(1))
   double hi = h[1], lo = l[1];
   for(int i = 2; i <= g_donN; i++)
   {
      if(h[i] > hi) hi = h[i];
      if(l[i] < lo) lo = l[i];
   }
   levelUp = hi;
   levelDn = lo;

   // TR 前缀和: S[i]=TR[1]+..+TR[i], TR[i]=max(h-l,|h-c[i+1]|,|l-c[i+1]|)
   int trMax = got - 2;                      // TR[i] 需要 c[i+1]
   if(trMax < 1 + InpATRPeriod)
      return true;                           // ATR 未预热: 通道有效但 gate 关
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
   atrPrev = (S[ap] - S[0]) / ap;            // ATR(bar1)=mean(TR[1..ap]) —— 研究口径
   if(atrPrev <= 0.0)
      return true;

   if(InpGateMode == 0)
   {
      gateOn = true;                         // 无闸门(R2变体)
      return true;
   }

   // 闸门: ATR[j], j=1..gateWin (窗口止于bar1且含bar1, 同引擎右移一根后语义)
   // valid计数 >= min_periods 才开闸(历史不足→关)
   int maxJ = MathMin(g_gateWin, trMax - ap + 1);   // j+ap-1 <= trMax
   if(maxJ < InpGateMinBars)
      return true;                           // gateOn=false
   double a[];
   ArrayResize(a, maxJ);
   for(int j = 1; j <= maxJ; j++)
      a[j - 1] = (S[j + ap - 1] - S[j - 1]) / ap;
   double q   = (InpGateMode == 2) ? 0.30 : 0.50;
   double ref = QuantileSorted(a, q);        // 注意: 就地排序
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
   PrintFormat("[v17] %s 失败: retcode=%u (%s)",
               what, trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| 成交事件: 记录入场bar(精确到deal时刻) + 出场PnL年度记账             |
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
      // 入场bar = 成交时刻所在bar(精确; bar内任意时点成交都归到该bar)
      g_entryBarTime = iTime(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_CURRENT, dtm));
      g_entryPrice   = price;
      g_nIn++;
      PrintFormat("[v17] ENTRY #%d %s @ %.2f vol=%.2f bar=%s → 持满%d根后开盘平仓",
                  g_nIn, (dtype == DEAL_TYPE_BUY ? "LONG" : "SHORT"),
                  price, vol, TimeToString(g_entryBarTime, TIME_DATE | TIME_MINUTES), g_hold);
   }
   else if(entryType == DEAL_ENTRY_OUT || entryType == DEAL_ENTRY_OUT_BY)
   {
      double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      double pnl;   // 平多(SELL)→(out-in); 平空(BUY)→(in-out); ×vol×contract
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

      PrintFormat("[v17] EXIT  #%d %s @ %.2f  pnl≈$%+.2f (%s)",
                  g_nOut, (dtype == DEAL_TYPE_SELL ? "close-L" : "close-S"),
                  price, pnl, rsn);
      if(g_entryBarTime != 0) g_entryBarTime = 0;
   }
}
//+------------------------------------------------------------------+
