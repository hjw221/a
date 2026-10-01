//+------------------------------------------------------------------+
//|                                              XauV3BalEnsEA.mq5   |
//|  v3bal_ens (XAUUSD, M5) — LightGBM/XGBoost 6成员×2方向 AUC加权集成  |
//|  ONNX端内推理, 无需Python。两种模式:                                |
//|    MODE_WF  : 逐月walk-forward复现 (2024-08~2026-07, 24折模型+每折阈值) |
//|    MODE_PROD: 生产模型 (2023-08~2026-07训练, 固定阈值)               |
//|  交易口径 1:1 对标 Python 管线 (xauusd_ml_v2):                       |
//|    信号=M5收盘特征→概率过阈值; 入场=市价(≈下一根M1开盘);              |
//|    TP=3.0×ATR288, SL=max(1.1429×ATR288, $0.30, 3×当前点差);          |
//|    6小时超时市价平; 平仓后冷却11分钟(出场M1K线+11); 单持仓。          |
//|  模型文件: 终端 Common\Files\<InpModelsDir>\{wf|prod}\ (见README_MT5) |
//|  v1.01 (2026-09-15 MT5 build5833实测反馈修复):                       |
//|   [1] OnnxSetInput/OutputShape补第2参数index=0 (v1.00少参数编译报错)  |
//|   [2] CSV改二进制读入自行解析: 兼容\n/\r\n/\r行尾+BOM (v1.00的       |
//|       FileReadString对\n-only CSV不可靠, 24折被读成"1折( ~ )"空表)   |
//|   [3] fold目录名补零(fold_00..09, v1.00拼成fold_0找不到)             |
//|   [4] 折表硬校验: 非24折/坏行直接INIT_FAILED并打印文件头hex诊断       |
//+------------------------------------------------------------------+
#property copyright "xauusd_ml_v2 pipeline (2026-09-15)"
#property version   "1.01"
#property description "v3bal_ens ONNX EA - replicates Python walk-forward backtest"

#include <Trade\Trade.mqh>

enum ENUM_EA_MODE { MODE_WF = 0, MODE_PROD = 1 };
enum ENUM_BARRIER_MODE { BARRIER_VIRTUAL = 0, BARRIER_SERVER = 1 };

//---------------------------- 输入 ---------------------------------
input ENUM_EA_MODE      InpMode          = MODE_WF;         // 模式: WF走查复现 / PROD生产
input ENUM_BARRIER_MODE InpBarrierMode   = BARRIER_VIRTUAL; // 障碍模式: 虚拟(对齐回测)/服务器TP SL
input double            InpLot           = 0.01;            // 手数(0.01手=1oz,对齐Python $口径)
input ulong             InpMagic         = 20260915;        // Magic number
input string            InpModelsDir     = "XauV3Ens";      // Common\Files 子目录名
input int               InpMinM5Bars     = 3000;            // M5回看根数(须>=2976)
input int               InpCooldownM1    = 10;              // 冷却M1根数(含出场K线共+11)
input int               InpHorizonM1     = 360;             // 超时M1根数(6小时)
input double            InpTpAtrMult     = 3.0;             // TP = 3.0 x ATR288
input double            InpSlAtrMult     = 1.1429;          // SL = 1.1429 x ATR288
input double            InpSlFloorUsd    = 0.30;            // SL绝对下限(美元)
input double            InpSpreadFloorMult = 3.0;           // SL下限 = 3 x 当前点差
input int               InpAtrWindow     = 288;             // ATR窗口(M5根)
input int               InpSlippagePts   = 50;              // 下单允许滑点(点)
input int               InpRunOrderForce = 0;               // OnnxRun参数顺序: 0自动探测
input double            InpSelfTestTol   = 0.003;           // 模型自检容差
input bool              InpDumpFeatures  = false;           // 导出特征CSV(对拍用)
input bool              InpVerbose       = true;            // 详细日志

#define N_FEAT 34
#define MAX_MEMBERS 16

//---------------------------- 全局 ---------------------------------
CTrade   trade;
double   g_nan = 0.0;

// 折索引 (WF)
int      g_nwf = 0;
string   g_wf_month[];
double   g_wf_thl[];
double   g_wf_ths[];
string   g_wf_fold[];

// 当前加载的集成
int      g_nm = 0;
long     g_h[MAX_MEMBERS];
double   g_w[MAX_MEMBERS];
int      g_mdir[MAX_MEMBERS];    // +1=long成员, -1=short成员
string   g_mfile[MAX_MEMBERS];
double   g_thr_l = 0.0, g_thr_s = 0.0;
string   g_cur_month = "";
bool     g_models_ok = false;
bool     g_selftest_ok = false;
bool     g_selftest_done = false;
int      g_run_order = 0;        // 0未知 1=输出在前 2=输入在前

// 持仓状态
datetime g_last_m5 = 0;
datetime g_entry_bar_min = 0;    // 入场M1K线open分钟
datetime g_cooldown_until = 0;
int      g_dir_open = 0;
double   g_tp_price = 0.0, g_sl_price = 0.0;
double   g_tp_d = 0.0, g_sl_d = 0.0;
datetime g_sig_time = 0;
double   g_sig_pl = 0.0, g_sig_ps = 0.0;
bool     g_had_pos = false;
datetime g_entry_time = 0;
double   g_entry_px = 0.0;
ulong    g_pos_id = 0;

// 统计
int      g_evals = 0, g_signals = 0, g_trades = 0;
int      g_warmwarn = 0, g_nomonth_warned = 0;

//+------------------------------------------------------------------+
//| 路径与文件工具                                                     |
//+------------------------------------------------------------------+
string PathCsv(const string rel)   { return InpModelsDir + "\\" + rel; }

// v1.01: 二进制读入整个文本文件并自行解析行 — 兼容 \n / \r\n / \r 行尾与UTF-8 BOM。
// 原因: MQL5 FileReadString(CSV模式)按\r\n行终止设计, 对Python写出的\n-only CSV
// 解析不可靠 (MT5实测: 24折wf_index.csv被读成"1折 ( ~ )"空表 -> 全部月份HOLD)。
bool ReadTextLines(const string rel, string &lines[])
{
   uchar data[];
   if(!ReadBytes(rel, data))
      return false;
   ArrayResize(lines, 0);
   int n = ArraySize(data);
   if(n == 0) return true;
   int start = 0;                                            // 去UTF-8 BOM
   if(n >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF)
      start = 3;
   string txt = CharArrayToString(data, start, n - start, CP_UTF8);
   StringReplace(txt, "\r\n", "\n");                         // 统一行尾为\n
   StringReplace(txt, "\r", "\n");
   string parts[];
   int k = StringSplit(txt, '\n', parts);
   for(int i = 0; i < k; i++)
   {
      string s = parts[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(s == "") continue;
      int m = ArraySize(lines);
      ArrayResize(lines, m + 1);
      lines[m] = s;
   }
   return true;
}

// v1.01: 诊断用 — 文件前64字节hex(排查BOM/编码/行尾); 读取失败给提示
string HexHead(const string rel)
{
   uchar data[];
   if(!ReadBytes(rel, data)) return "(读取失败)";
   int n = ArraySize(data);
   if(n > 64) n = 64;
   string s = "";
   for(int i = 0; i < n; i++)
      s += StringFormat("%02X ", data[i]);
   return StringFormat("size=%d字节 head=%s", ArraySize(data), s);
}

// v1.01: "YYYY-MM"格式校验
bool IsValidMonthStr(const string mon)
{
   if(StringLen(mon) != 7) return false;
   if(StringGetCharacter(mon, 4) != '-') return false;
   for(int i = 0; i < 7; i++)
   {
      if(i == 4) continue;
      ushort ch = StringGetCharacter(mon, i);
      if(ch < '0' || ch > '9') return false;
   }
   int mm = (int)StringToInteger(StringSubstr(mon, 5, 2));
   return (mm >= 1 && mm <= 12);
}

bool ReadBytes(const string rel, uchar &data[])
{
   int fh = FileOpen(PathCsv(rel), FILE_READ | FILE_BIN | FILE_COMMON);
   if(fh == INVALID_HANDLE)
      fh = FileOpen(PathCsv(rel), FILE_READ | FILE_BIN);
   if(fh == INVALID_HANDLE) return false;
   int sz = (int)FileSize(fh);
   ArrayResize(data, sz);
   FileReadArray(fh, data, 0, sz);
   FileClose(fh);
   return true;
}

int OpenCsvAppend(const string rel)
{
   int fh = FileOpen(PathCsv(rel), FILE_READ | FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(fh == INVALID_HANDLE)
      fh = FileOpen(PathCsv(rel), FILE_READ | FILE_WRITE | FILE_CSV, ',');
   if(fh == INVALID_HANDLE) return INVALID_HANDLE;
   FileSeek(fh, 0, SEEK_END);
   return fh;
}

string FmtG(double v) { return StringFormat("%.10g", v); }

//+------------------------------------------------------------------+
//| ONNX 工具                                                         |
//+------------------------------------------------------------------+
bool LoadOnnxModel(const string rel, long &handle)
{
   uchar data[];
   if(!ReadBytes(rel, data))
   {
      PrintFormat("[onnx] 文件缺失: %s (安装到 Common\\Files\\%s)", PathCsv(rel), InpModelsDir);
      return false;
   }
   ulong flags = InpVerbose ? ONNX_DEBUG_LOGS : 0;
   handle = OnnxCreateFromBuffer(data, flags);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("[onnx] OnnxCreateFromBuffer失败 %s err=%d (终端版本需>=3980)",
                  rel, GetLastError());
      return false;
   }
   ulong in_shape[]  = {1, N_FEAT};
   ulong out_shape[] = {1, 2};
   if(!OnnxSetInputShape(handle, 0, in_shape))
      PrintFormat("[onnx] SetInputShape警告 %s err=%d", rel, GetLastError());
   if(!OnnxSetOutputShape(handle, 0, out_shape))
      PrintFormat("[onnx] SetOutputShape警告 %s err=%d", rel, GetLastError());
   return true;
}

// 单行推理: 输入34特征, 返回P(class=1)。参数顺序自动探测。
bool RunOnnxOne(long h, const float &x[], double &p1)
{
   float probs[2];
   ResetLastError();
   if(g_run_order != 2)
   {
      if(OnnxRun(h, 0, probs, x))     // 尝试: 输出在前
      {
         if(g_run_order == 0) g_run_order = 1;
         p1 = (double)probs[1];
         return true;
      }
      if(g_run_order == 1) { PrintFormat("[onnx] run失败(输出在前) err=%d", GetLastError()); return false; }
   }
   ResetLastError();
   if(g_run_order != 1)
   {
      if(OnnxRun(h, 0, x, probs))     // 尝试: 输入在前
      {
         if(g_run_order == 0) g_run_order = 2;
         p1 = (double)probs[1];
         return true;
      }
      if(g_run_order == 2) { PrintFormat("[onnx] run失败(输入在前) err=%d", GetLastError()); return false; }
   }
   return false;
}

// 加权集成: 返回 (pl, ps)
bool EnsembleProbs(const double &f[], double &pl, double &ps)
{
   pl = 0.0; ps = 0.0;
   float x[N_FEAT];
   for(int k = 0; k < N_FEAT; k++)
      x[k] = (float)f[k];
   for(int m = 0; m < g_nm; m++)
   {
      double p1;
      if(!RunOnnxOne(g_h[m], x, p1))
      {
         PrintFormat("[ens] 成员%d(%s)推理失败", m, g_mfile[m]);
         return false;
      }
      if(g_mdir[m] > 0) pl += g_w[m] * p1;
      else              ps += g_w[m] * p1;
   }
   return true;
}

//+------------------------------------------------------------------+
//| 释放当前集成                                                       |
//+------------------------------------------------------------------+
void ReleaseEnsemble()
{
   for(int m = 0; m < g_nm; m++)
      if(g_h[m] != INVALID_HANDLE) OnnxRelease(g_h[m]);
   g_nm = 0;
   g_models_ok = false;
   g_selftest_ok = false;
   g_selftest_done = false;
}

//+------------------------------------------------------------------+
//| 从 members.csv + selftest.csv 加载一套集成                          |
//+------------------------------------------------------------------+
bool LoadEnsemble(const string folder_rel, const double thl, const double ths)
{
   ReleaseEnsemble();
   // ---- members.csv: direction,kind,file,weight (v1.01二进制解析) ----
   string mlines[];
   if(!ReadTextLines(folder_rel + "\\members.csv", mlines) || ArraySize(mlines) == 0)
   {
      PrintFormat("[load] 缺/空 %s\\members.csv (诊断: %s)", folder_rel,
                  HexHead(folder_rel + "\\members.csv"));
      return false;
   }
   for(int li = 0; li < ArraySize(mlines) && g_nm < MAX_MEMBERS; li++)
   {
      string fld[];
      if(StringSplit(mlines[li], ',', fld) < 4) continue;
      string d = fld[0];
      string file = fld[2];
      long h;
      if(!LoadOnnxModel(folder_rel + "\\" + file, h))
         return false;
      g_h[g_nm] = h;
      g_w[g_nm] = StringToDouble(fld[3]);
      g_mdir[g_nm] = (d == "long") ? 1 : -1;
      g_mfile[g_nm] = file;
      g_nm++;
   }
   if(g_nm == 0)
   {
      PrintFormat("[load] %s 无成员", folder_rel);
      return false;
   }
   g_thr_l = thl;
   g_thr_s = ths;
   // ---- selftest.csv: expected_pl,expected_ps,f1..f34 (v1.01二进制解析) ----
   double epl = 0, eps_ = 0;
   double f34[N_FEAT];
   bool have_st = false;
   string slines[];
   if(ReadTextLines(folder_rel + "\\selftest.csv", slines) && ArraySize(slines) >= 1)
   {
      string sf[];
      if(StringSplit(slines[0], ',', sf) >= 2 + N_FEAT)
      {
         epl = StringToDouble(sf[0]);
         eps_ = StringToDouble(sf[1]);
         for(int k = 0; k < N_FEAT; k++)
            f34[k] = StringToDouble(sf[2 + k]);
         have_st = true;
      }
   }
   g_models_ok = true;
   double wsum_l = 0, wsum_s = 0;
   for(int m = 0; m < g_nm; m++)
      if(g_mdir[m] > 0) wsum_l += g_w[m]; else wsum_s += g_w[m];
   PrintFormat("[load] %s: %d个成员 (long权重和%.4f / short权重和%.4f), thr L=%.4f S=%.4f",
               folder_rel, g_nm, wsum_l, wsum_s, g_thr_l, g_thr_s);
   // ---- 自检 ----
   if(have_st)
   {
      double pl, ps;
      if(EnsembleProbs(f34, pl, ps))
      {
         double dl = MathAbs(pl - epl), ds = MathAbs(ps - eps_);
         g_selftest_ok = (dl <= InpSelfTestTol && ds <= InpSelfTestTol);
         g_selftest_done = true;
         PrintFormat("[self-test] pL=%.4f(期望%.4f Δ%.1e) pS=%.4f(期望%.4f Δ%.1e) => %s",
                     pl, epl, dl, ps, eps_, ds, g_selftest_ok ? "PASS" : "FAIL !!");
         if(!g_selftest_ok)
            Print("[self-test] FAIL: 模型绑定/特征顺序有问题, 本套模型不交易!");
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| v1.01: 加载WF折索引(二进制解析+硬校验), 坏表直接INIT_FAILED        |
//+------------------------------------------------------------------+
bool LoadWfIndex()
{
   string lines[];
   if(!ReadTextLines("wf\\wf_index.csv", lines))
   {
      PrintFormat("[init] 缺 wf\\wf_index.csv — 请把mt5_package/models整目录复制到 Common\\Files\\%s\\ (后备: MQL5\\Files\\%s\\)",
                  InpModelsDir, InpModelsDir);
      return false;
   }
   ArrayResize(g_wf_month, 0); ArrayResize(g_wf_thl, 0);
   ArrayResize(g_wf_ths, 0);   ArrayResize(g_wf_fold, 0);
   int bad = 0;
   for(int li = 0; li < ArraySize(lines); li++)
   {
      string fld[];
      if(StringSplit(lines[li], ',', fld) < 4) { bad++; continue; }
      string fold = fld[0];
      string mon  = fld[1];
      double tl = StringToDouble(fld[2]);
      double ts = StringToDouble(fld[3]);
      if(!IsValidMonthStr(mon) || !(tl > 0.0 && tl < 1.0) || !(ts > 0.0 && ts < 1.0))
      {
         bad++;
         continue;
      }
      int n = ArraySize(g_wf_month);
      ArrayResize(g_wf_month, n + 1); ArrayResize(g_wf_thl, n + 1);
      ArrayResize(g_wf_ths, n + 1);   ArrayResize(g_wf_fold, n + 1);
      g_wf_month[n] = mon; g_wf_thl[n] = tl;
      g_wf_ths[n] = ts;   g_wf_fold[n] = fold;
   }
   int nwf = ArraySize(g_wf_month);
   if(nwf != 24 || bad > 0)
   {
      PrintFormat("[init] wf_index.csv 解析异常: 有效折=%d 坏行=%d (v3bal_ens期望24折)。文件诊断: %s",
                  nwf, bad, HexHead("wf\\wf_index.csv"));
      if(nwf > 0)
         PrintFormat("[init] 已解析首末月份: %s ~ %s", g_wf_month[0], g_wf_month[nwf - 1]);
      return false;
   }
   PrintFormat("[init] WF索引: %d折 (%s ~ %s)", nwf, g_wf_month[0], g_wf_month[nwf - 1]);
   return true;
}

//+------------------------------------------------------------------+
//| 初始化                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_nan = MathSqrt(-1.0);
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePts);
   trade.SetTypeFillingBySymbol(_Symbol);
   if(_Point != 0.001)
      PrintFormat("[warn] 当前品种point=%.5f (训练数据为0.001三位报价)。点差成本语义将与回测数据不同, 结果仅供参考。",
                  _Point);
   PrintFormat("[init] 账户币种=%s point=%.5f 合约规模=%.0f — USC美分账户: 报表金额除以100≈Python美元口径(合约=100oz时)",
               AccountInfoString(ACCOUNT_CURRENCY), _Point,
               SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));
   PrintFormat("[init] 模式=%s 障碍=%s lot=%.2f ATR窗口=%d 回看M5=%d根",
               InpMode == MODE_WF ? "WF走查复现(2024-08~2026-07)" : "PROD生产模型",
               InpBarrierMode == BARRIER_VIRTUAL ? "虚拟(对齐Python回测)" : "服务器TP/SL",
               InpLot, InpAtrWindow, InpMinM5Bars);
   if(InpMinM5Bars < 2976)
      Print("[warn] InpMinM5Bars<2976, squeeze特征将缺NaN -> HOLD");
   if(InpRunOrderForce != 0)
      g_run_order = InpRunOrderForce;
   if(InpMode == MODE_WF)
   {
      if(!LoadWfIndex())
         return INIT_FAILED;
      g_cur_month = "";   // 首个M5收盘时按月加载
   }
   else
   {
      string plines[];
      if(!ReadTextLines("prod\\index.csv", plines) || ArraySize(plines) == 0)
      {
         PrintFormat("[init] 缺/空 prod\\index.csv — 请把mt5_package/models整目录复制到 Common\\Files\\%s\\", InpModelsDir);
         return INIT_FAILED;
      }
      string pf[];
      if(StringSplit(plines[0], ',', pf) < 4)
      {
         PrintFormat("[init] prod\\index.csv 解析失败: 行='%s' (诊断: %s)", plines[0], HexHead("prod\\index.csv"));
         return INIT_FAILED;
      }
      if(!LoadEnsemble("prod", StringToDouble(pf[1]), StringToDouble(pf[2])))
         return INIT_FAILED;
      PrintFormat("[init] PROD: %s 训练窗%s", pf[0], pf[3]);
      g_cur_month = "PROD";
   }
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   ReleaseEnsemble();
}

//+------------------------------------------------------------------+
//| 按月切换折 (WF模式)                                                |
//+------------------------------------------------------------------+
void EnsureFoldForMonth(const string mon)
{
   if(InpMode != MODE_WF) return;
   if(mon == g_cur_month) return;
   g_cur_month = mon;
   int found = -1;
   for(int k = 0; k < ArraySize(g_wf_month); k++)
      if(g_wf_month[k] == mon) { found = k; break; }
   if(found < 0)
   {
      ReleaseEnsemble();
      if(g_nomonth_warned < 3 || InpVerbose)
      {
         int nw = ArraySize(g_wf_month);
         string rng = (nw > 0) ? (g_wf_month[0] + "~" + g_wf_month[nw - 1]) : "空表";
         PrintFormat("[wf] %s 不在走查范围(%s), HOLD", mon, rng);
      }
      g_nomonth_warned++;
      return;
   }
   string fold_str = g_wf_fold[found];
   if(StringLen(fold_str) < 2) fold_str = "0" + fold_str;   // v1.01: 目录名fold_00..fold_23补零
   string folder = "wf\\fold_" + fold_str;
   if(LoadEnsemble(folder, g_wf_thl[found], g_wf_ths[found]))
      PrintFormat("[wf] 切换到折%s = %s", fold_str, mon);
   else
      PrintFormat("[wf] 折%s(%s)模型加载失败 — 本月HOLD! 检查 Common\\Files\\%s\\%s\\ 完整性",
                  fold_str, mon, InpModelsDir, folder);
}

//+------------------------------------------------------------------+
//| 主循环                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageOpenPosition();
   // 新M5K线 -> 评估刚收盘K线
   datetime t0 = iTime(_Symbol, PERIOD_M5, 0);
   if(t0 == 0) return;
   if(t0 == g_last_m5) return;
   g_last_m5 = t0;
   EvaluateClosedBar();
}

//+------------------------------------------------------------------+
//| 评估刚收盘的M5K线: 特征->集成概率->信号->入场                        |
//+------------------------------------------------------------------+
void EvaluateClosedBar()
{
   int need = InpMinM5Bars + 1;
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int n = CopyRates(_Symbol, PERIOD_M5, 0, need, rates);
   if(n < need)
   {
      g_warmwarn++;
      if(g_warmwarn % 50 == 1)
         PrintFormat("[warmup] M5历史不足 %d/%d根, 等待...", n, need);
      return;
   }
   int i = n - 2;                    // 刚收盘K线(n-1是进行中)
   double o[], h[], l[], c[], v[], sp[];
   ArrayResize(o, n); ArrayResize(h, n); ArrayResize(l, n);
   ArrayResize(c, n); ArrayResize(v, n); ArrayResize(sp, n);
   for(int k = 0; k < n; k++)
   {
      o[k] = rates[k].open; h[k] = rates[k].high; l[k] = rates[k].low;
      c[k] = rates[k].close; v[k] = (double)rates[k].tick_volume;
      sp[k] = (double)rates[k].spread;   // 点数
   }
   double f[N_FEAT];
   double atr = 0.0;
   if(!ComputeFeatures(o, h, l, c, v, sp, rates, i, f, atr))
   {
      if(InpVerbose) Print("[feat] 特征含NaN(数据不足/异常K线), 本K线HOLD");
      return;
   }
   g_evals++;
   if(InpDumpFeatures) DumpFeatureRow(rates[i].time, f);

   // ---- 按月切折 ----
   MqlDateTime st; TimeToStruct(rates[i].time, st);
   string mon = StringFormat("%04d-%02d", st.year, st.mon);
   EnsureFoldForMonth(mon);
   if(!g_models_ok) return;                          // 无模型 -> HOLD
   if(g_selftest_done && !g_selftest_ok) return;      // 自检失败 -> HOLD (无自检文件则放行)

   // ---- 集成概率 ----
   double pl, ps;
   if(!EnsembleProbs(f, pl, ps)) return;

   // ---- 信号规则 (与backtest.build_signals一致) ----
   bool lsig = pl > g_thr_l, ssig = ps > g_thr_s;
   int sig = 0;
   if(lsig && ssig)       sig = ((pl - g_thr_l) >= (ps - g_thr_s)) ? 1 : -1;
   else if(lsig)          sig = 1;
   else if(ssig)          sig = -1;
   if(sig == 0) return;
   g_signals++;

   // ---- 可交易性 ----
   ulong ticket;
   if(HasOurPosition(ticket)) return;                    // 单持仓
   if(TimeCurrent() < g_cooldown_until) return;          // 冷却
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0 || bid <= 0 || ask - bid < 0) return;
   double spread_usd = MathMax(ask - bid, _Point);       // python: max(sp_pts,1)*point

   // ---- ATR自适应障碍 ----
   double tp_d = InpTpAtrMult * atr;
   double sl_d = MathMax(InpSlAtrMult * atr, MathMax(InpSlFloorUsd, InpSpreadFloorMult * spread_usd));
   if(tp_d <= 0 || sl_d <= 0 || atr <= 0) return;

   // ---- 下单 ----
   g_sig_time = rates[i].time;
   g_sig_pl = pl; g_sig_ps = ps;
   g_tp_d = tp_d; g_sl_d = sl_d;
   bool okr;
   if(sig > 0)
   {
      double tpp = (InpBarrierMode == BARRIER_SERVER) ? ask + tp_d : 0.0;
      double slp = (InpBarrierMode == BARRIER_SERVER) ? ask - sl_d : 0.0;
      okr = trade.Buy(InpLot, _Symbol, 0.0, slp, tpp, "v3ens");
   }
   else
   {
      double tpp = (InpBarrierMode == BARRIER_SERVER) ? bid - tp_d : 0.0;
      double slp = (InpBarrierMode == BARRIER_SERVER) ? bid + sl_d : 0.0;
      okr = trade.Sell(InpLot, _Symbol, 0.0, slp, tpp, "v3ens");
   }
   if(!okr || trade.ResultRetcode() != TRADE_RETCODE_DONE)
   {
      PrintFormat("[order] 失败 ret=%d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return;
   }
   g_dir_open = sig;
   ulong tk2;
   if(HasOurPosition(tk2))
   {
      g_entry_bar_min = (datetime)(((long)PositionGetInteger(POSITION_TIME)) / 60 * 60);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      g_entry_time = (datetime)PositionGetInteger(POSITION_TIME);
      g_entry_px = entry;
      g_pos_id = (ulong)PositionGetInteger(POSITION_IDENTIFIER);
      if(InpBarrierMode == BARRIER_VIRTUAL)
      {
         if(sig > 0) { g_tp_price = entry + tp_d; g_sl_price = entry - sl_d; }
         else        { g_tp_price = entry - tp_d; g_sl_price = entry + sl_d; }
      }
      g_trades++;
      PrintFormat("[trade #%d] %s @%.2f pL=%.4f/%.4f pS=%.4f/%.4f ATR=%.3f TP$%.2f SL$%.2f",
                  g_trades, sig > 0 ? "LONG" : "SHORT", entry,
                  pl, g_thr_l, ps, g_thr_s, atr, tp_d, sl_d);
   }
}

//+------------------------------------------------------------------+
//| 持仓管理: 虚拟障碍 + 6h超时                                        |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   ulong ticket;
   if(!HasOurPosition(ticket))
   {
      if(g_had_pos) OnExternalClose();   // 服务器TP/SL触发的外部平仓
      g_had_pos = false;
      return;
   }
   g_had_pos = true;
   datetime now = TimeCurrent();
   // ---- 超时 (Python: 入场M1K线open + 361分钟 = 第360根M1收盘) ----
   if(now >= g_entry_bar_min + (datetime)(InpHorizonM1 + 1) * 60)
   {
      CloseOurs("TIMEOUT", ticket);
      return;
   }
   if(InpBarrierMode == BARRIER_VIRTUAL && now >= g_entry_bar_min + 60)
   // 障碍检查从入场后第2根M1开始 (Python: j从e+1起, 入场K线本身不检查)
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(g_dir_open > 0)
      {
         if(bid <= g_sl_price) { CloseOurs("SL", ticket); return; }
         if(bid >= g_tp_price) { CloseOurs("TP", ticket); return; }
      }
      else
      {
         if(ask >= g_sl_price) { CloseOurs("SL", ticket); return; }
         if(ask <= g_tp_price) { CloseOurs("TP", ticket); return; }
      }
   }
}

// 平仓并记录 + 冷却
void CloseOurs(const string reason, ulong ticket)
{
   double exit_px = 0.0;
   if(g_dir_open > 0) exit_px = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   else               exit_px = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(!trade.PositionClose(ticket))
   {
      PrintFormat("[close] 失败 ret=%d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return;
   }
   g_had_pos = false;   // 防止下一tick误触发OnExternalClose双记日志
   datetime now = TimeCurrent();
   // 冷却: Python next_free = 出场M1K线 + 11根 => t_xb + (cooldown+1)分钟
   datetime exit_bar_min;
   if(reason == "TIMEOUT") exit_bar_min = g_entry_bar_min + (datetime)InpHorizonM1 * 60;
   else                    exit_bar_min = (datetime)(((long)now) / 60 * 60);
   g_cooldown_until = exit_bar_min + (datetime)(InpCooldownM1 + 1) * 60;
   // 交易日志
   double pnl = 0.0;
   ulong pos_id = g_pos_id;   // 位置ID(netting/hedging通用)
   if(HistorySelectByPosition(pos_id))
   {
      int nd = HistoryDealsTotal();
      for(int k = 0; k < nd; k++)
      {
         ulong dk = HistoryDealGetTicket(k);
         if(dk > 0)
            pnl += HistoryDealGetDouble(dk, DEAL_PROFIT)
                 + HistoryDealGetDouble(dk, DEAL_SWAP)
                 + HistoryDealGetDouble(dk, DEAL_COMMISSION);
      }
   }
   PrintFormat("[exit] %s @%.2f 冷却至%s PnL$%.2f", reason, exit_px,
               TimeToString(g_cooldown_until, TIME_DATE | TIME_MINUTES), pnl);
   WriteTradeRow(reason, exit_px, pnl);
}

//+------------------------------------------------------------------+
//| 持仓查询(magic过滤)                                                |
//+------------------------------------------------------------------+
bool HasOurPosition(ulong &ticket)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         (ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic)
      {
         ticket = tk;
         return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| 服务器模式: TP/SL被服务器平仓后的补记 (冷却+日志)                    |
//+------------------------------------------------------------------+
void OnExternalClose()
{
   datetime now = TimeCurrent();
   string outcome = "CLOSE";
   double exit_px = 0.0, pnl = 0.0;
   if(HistorySelectByPosition(g_pos_id))
   {
      int nd = HistoryDealsTotal();
      for(int k = 0; k < nd; k++)
      {
         ulong dk = HistoryDealGetTicket(k);
         if(dk == 0) continue;
         pnl += HistoryDealGetDouble(dk, DEAL_PROFIT)
              + HistoryDealGetDouble(dk, DEAL_SWAP)
              + HistoryDealGetDouble(dk, DEAL_COMMISSION);
         long reason = HistoryDealGetInteger(dk, DEAL_REASON);
         if(reason == DEAL_REASON_TP)      { outcome = "TP";  exit_px = HistoryDealGetDouble(dk, DEAL_PRICE); }
         else if(reason == DEAL_REASON_SL) { outcome = "SL";  exit_px = HistoryDealGetDouble(dk, DEAL_PRICE); }
         else if(HistoryDealGetInteger(dk, DEAL_ENTRY) == DEAL_ENTRY_OUT)
                                          { outcome = "CLOSE"; exit_px = HistoryDealGetDouble(dk, DEAL_PRICE); }
      }
   }
   g_cooldown_until = (datetime)(((long)now) / 60 * 60) + (datetime)(InpCooldownM1 + 1) * 60;
   PrintFormat("[exit] %s(服务器) @%.2f 冷却至%s PnL$%.2f", outcome, exit_px,
               TimeToString(g_cooldown_until, TIME_DATE | TIME_MINUTES), pnl);
   WriteTradeRow(outcome, exit_px, pnl);
}

//+------------------------------------------------------------------+
//| ============ 特征引擎 (1:1移植 features_v3.py, 34特征) ============ |
//+------------------------------------------------------------------+
string FEAT_NAMES[N_FEAT] =
{
   "atr_ratio", "tr_over_atr", "squeeze", "range_ratio",
   "momn_12", "momn_48", "momn_96", "momn_288", "mom_acc",
   "er_24", "er_96", "er_288", "slope_96", "slope_288",
   "dist_hi_288", "dist_lo_288", "pos_in_range",
   "bars_since_hi96", "bars_since_lo96", "run_len",
   "c_dist_s4h", "c_dist_d1", "vol_z_288", "vol_ratio_24_288",
   "body", "upper_shadow", "lower_shadow",
   "hour_sin", "hour_cos", "dow_sin", "dow_cos",
   "session_london", "session_ny", "spread_rel"
};

// 真实波幅: max(h-l, |h-前收|, |l-前收|)
double TrueRange(const double &h[], const double &l[], const double &c[], const int j)
{
   if(j <= 0) return h[j] - l[j];
   double a = h[j] - l[j];
   double b = MathAbs(h[j] - c[j - 1]);
   double d = MathAbs(l[j] - c[j - 1]);
   return MathMax(a, MathMax(b, d));
}

// Kaufman效率比: |净位移| / 路径总长 (NaN→g_nan)
double ErVal(const double &c[], const int i, const int p)
{
   double num = MathAbs(c[i] - c[i - p]);
   double den = 0.0;
   for(int j = i - p + 1; j <= i; j++)
      den += MathAbs(c[j] - c[j - 1]);
   if(den == 0.0) return g_nan;
   return num / den;
}

// n根线性回归斜率 (绝对索引可平移, 结果不变)
double SlopeVal(const double &c[], const int i, const int nn)
{
   double Sy = 0.0, Sky = 0.0;
   for(int k = i - nn + 1; k <= i; k++)
   {
      Sy  += c[k];
      Sky += (double)k * c[k];
   }
   double xbar  = (double)(i - nn + 1) + (nn - 1) / 2.0;
   double denom = nn * ((double)nn * nn - 1.0) / 12.0;
   return (Sky - xbar * Sy) / denom;
}

// 距最近nn根内最高/最低过了多少根 (等值取最新, 对齐单调队列语义)
double BarsSince(const double &a[], const int i, const int nn, const bool is_max)
{
   int best = i;
   double bv = a[i];
   for(int k = i - 1; k >= i - nn + 1; k--)
   {
      if(is_max ? (a[k] > bv) : (a[k] < bv))
      {
         bv = a[k];
         best = k;
      }
   }
   return (double)(i - best);
}

// 中位数 (升序数组)
double MedianSorted(const double &a[], const int cnt)
{
   if(cnt <= 0) return g_nan;
   if(cnt % 2 == 1) return a[cnt / 2];
   return (a[cnt / 2 - 1] + a[cnt / 2]) / 2.0;
}

//+------------------------------------------------------------------+
//| 计算34特征 (窗口全部含当前K线i, 与pandas rolling一致)               |
//| 返回false=有NaN(数据不足/异常) => HOLD (与predict_live同策略)        |
//+------------------------------------------------------------------+
bool ComputeFeatures(const double &o[], const double &h[], const double &l[], const double &c[],
                     const double &v[], const double &sp[], const MqlRates &rates[],
                     const int i, double &f[], double &atr)
{
   if(i < 2975) return false;                    // squeeze最深回看 2880+96-1
   // ---- [1] TR / ATR ----
   double tr_sum288 = 0.0, tr_sum24 = 0.0;
   for(int j = i - 287; j <= i; j++) tr_sum288 += TrueRange(h, l, c, j);
   for(int j = i - 23;  j <= i; j++) tr_sum24  += TrueRange(h, l, c, j);
   atr = tr_sum288 / 288.0;
   if(!(atr > 0.0) || !MathIsValidNumber(atr)) return false;
   f[0] = (tr_sum24 / 24.0) / atr;                       // atr_ratio
   f[1] = TrueRange(h, l, c, i) / atr;                   // tr_over_atr
   // ---- [2] squeeze = 4*std96(c) / median2880(bb_w) ----
   int j0 = i - 2879;
   double bbw[];
   ArrayResize(bbw, 2880);
   double s = 0.0, s2 = 0.0;
   for(int k = j0 - 95; k <= j0; k++) { s += c[k]; s2 += c[k] * c[k]; }
   for(int j = j0; j <= i; j++)
   {
      double varr = (s2 - s * s / 96.0) / 95.0;         // 样本std (ddof=1)
      if(varr < 0.0) varr = 0.0;
      bbw[j - j0] = 4.0 * MathSqrt(varr);
      if(j < i)
      {
         s  += c[j + 1] - c[j + 1 - 96];
         s2 += c[j + 1] * c[j + 1] - c[j + 1 - 96] * c[j + 1 - 96];
      }
   }
   double tmp[];
   ArrayResize(tmp, 2880);
   ArrayCopy(tmp, bbw, 0, 0, 2880);
   ArraySort(tmp);
   double med_bbw = MedianSorted(tmp, 2880);
   if(med_bbw == 0.0 || !MathIsValidNumber(med_bbw)) return false;
   f[2] = bbw[2879] / med_bbw;                           // squeeze
   // ---- [3] range_ratio ----
   double maxh96 = -DBL_MAX, minl96 = DBL_MAX, maxh288 = -DBL_MAX, minl288 = DBL_MAX;
   for(int k = i - 95; k <= i; k++)
   {
      if(h[k] > maxh96) maxh96 = h[k];
      if(l[k] < minl96) minl96 = l[k];
   }
   for(int k = i - 287; k <= i; k++)
   {
      if(h[k] > maxh288) maxh288 = h[k];
      if(l[k] < minl288) minl288 = l[k];
   }
   double rng288 = maxh288 - minl288;
   if(rng288 == 0.0) return false;
   f[3] = (maxh96 - minl96) / rng288;
   // ---- [4] ATR归一动量 + 加速度 ----
   f[4] = (c[i] - c[i - 12])  / atr;
   f[5] = (c[i] - c[i - 48])  / atr;
   f[6] = (c[i] - c[i - 96])  / atr;
   f[7] = (c[i] - c[i - 288]) / atr;
   f[8] = f[4] - 0.25 * f[5];
   // ---- [5] 效率比 + 回归斜率 ----
   f[9]  = ErVal(c, i, 24);
   f[10] = ErVal(c, i, 96);
   f[11] = ErVal(c, i, 288);
   f[12] = SlopeVal(c, i, 96)  / atr;
   f[13] = SlopeVal(c, i, 288) / atr;
   // ---- [6] 突破结构 (前288根不含当前) ----
   double hp = -DBL_MAX, lp = DBL_MAX;
   for(int k = i - 288; k <= i - 1; k++)
   {
      if(h[k] > hp) hp = h[k];
      if(l[k] < lp) lp = l[k];
   }
   f[14] = (c[i] - hp) / atr;
   f[15] = (c[i] - lp) / atr;
   f[16] = (c[i] - minl288) / rng288;                    // pos_in_range
   f[17] = BarsSince(h, i, 96, true);
   f[18] = BarsSince(l, i, 96, false);
   // ---- [7] run_len: 同向连续收盘(平盘沿用方向且计数+1), ±10截断归一 ----
   int dir = 0, cnt = 0;
   for(int j = 1; j <= i; j++)
   {
      double d = c[j] - c[j - 1];
      if(d > 0)      { if(dir == 1)  cnt++; else { dir = 1;  cnt = 1; } }
      else if(d < 0) { if(dir == -1) cnt++; else { dir = -1; cnt = 1; } }
      else           { cnt++; }                           // 平盘: 同组继续计数
   }
   f[19] = dir * (double)MathMin(cnt, 10) / 10.0;
   // ---- [8] 超涨超跌 (SMA96/288) ----
   double sma96 = 0.0, sma288 = 0.0;
   for(int k = i - 95; k <= i; k++)  sma96  += c[k];
   for(int k = i - 287; k <= i; k++) sma288 += c[k];
   f[20] = (c[i] - sma96 / 96.0) / atr;
   f[21] = (c[i] - sma288 / 288.0) / atr;
   // ---- [9] 量能 ----
   double v24 = 0.0, v288 = 0.0, v2882 = 0.0;
   for(int k = i - 23; k <= i; k++) v24 += v[k];
   for(int k = i - 287; k <= i; k++) { v288 += v[k]; v2882 += v[k] * v[k]; }
   double mv = v288 / 288.0;
   double vv = (v2882 - v288 * v288 / 288.0) / 287.0;
   if(vv < 0.0) vv = 0.0;
   double sdv = MathSqrt(vv);
   f[22] = (sdv > 0.0) ? (v[i] - mv) / sdv : g_nan;
   f[23] = (mv  > 0.0) ? (v24 / 24.0) / mv : g_nan;
   // ---- [10] K线形态 ----
   double rng1 = h[i] - l[i];
   if(rng1 == 0.0) return false;
   double mx_oc = MathMax(o[i], c[i]), mn_oc = MathMin(o[i], c[i]);
   f[24] = (c[i] - o[i]) / rng1;
   f[25] = (h[i] - mx_oc) / rng1;
   f[26] = (mn_oc - l[i]) / rng1;
   // ---- [11] 时间结构 (服务器时间, 与训练CSV同源) ----
   MqlDateTime stt;
   TimeToStruct(rates[i].time, stt);
   double hr = stt.hour + stt.min / 60.0;
   f[27] = MathSin(hr * 2.0 * M_PI / 24.0);
   f[28] = MathCos(hr * 2.0 * M_PI / 24.0);
   int dow_pd = (stt.day_of_week + 6) % 7;               // MQL5 0=周日 -> pandas 0=周一
   f[29] = MathSin(dow_pd * 2.0 * M_PI / 7.0);
   f[30] = MathCos(dow_pd * 2.0 * M_PI / 7.0);
   f[31] = (stt.hour >= 7  && stt.hour < 13) ? 1.0 : 0.0; // 伦敦时段
   f[32] = (stt.hour >= 13 && stt.hour < 21) ? 1.0 : 0.0; // 纽约时段
   // ---- [12] spread_rel: 当前点差 / 非零点差288根中位数(≥144个) ----
   if(sp[i] <= 0.0) return false;                        // 当前点差0 -> NaN -> HOLD
   {
      double tmps[];
      ArrayResize(tmps, 288);
      int cnts = 0;
      for(int k = i - 287; k <= i; k++)
         if(sp[k] > 0.0) tmps[cnts++] = sp[k];
      if(cnts < 144) return false;
      ArrayResize(tmps, cnts);
      ArraySort(tmps);
      double med_sp = MedianSorted(tmps, cnts);
      if(med_sp <= 0.0) return false;
      f[33] = sp[i] / med_sp;
   }
   // ---- 终检: 任一NaN -> HOLD ----
   for(int k = 0; k < N_FEAT; k++)
      if(!MathIsValidNumber(f[k])) return false;
   return true;
}

//+------------------------------------------------------------------+
//| 特征导出 (对拍: 与 reference/features_reference_python.csv 逐行diff)|
//+------------------------------------------------------------------+
void DumpFeatureRow(const datetime t, const double &f[])
{
   static bool written = false;
   int fh = OpenCsvAppend("features_EA.csv");
   if(fh == INVALID_HANDLE) return;
   if(!written)
   {
      string head = "time";
      for(int k = 0; k < N_FEAT; k++) head += "," + FEAT_NAMES[k];
      FileWrite(fh, head);
      written = true;
   }
   string line = TimeToString(t, TIME_DATE | TIME_MINUTES);
   for(int k = 0; k < N_FEAT; k++) line += "," + FmtG(f[k]);
   FileWrite(fh, line);
   FileClose(fh);
}

//+------------------------------------------------------------------+
//| 交易日志 (对拍: reference/trades_v3bal_ens_python.csv)              |
//+------------------------------------------------------------------+
void WriteTradeRow(const string outcome, const double exit_px, const double pnl)
{
   static bool written = false;
   int fh = OpenCsvAppend("trades_EA.csv");
   if(fh == INVALID_HANDLE) return;
   if(!written)
   {
      FileWrite(fh, "signal_time,dir,entry_time,entry_px,exit_time,exit_px,outcome,pnl_usd,tp_d,sl_d,prob");
      written = true;
   }
   string line = TimeToString(g_sig_time, TIME_DATE | TIME_MINUTES) + "," +
                 (g_dir_open > 0 ? "long" : "short") + "," +
                 TimeToString(g_entry_time, TIME_DATE | TIME_MINUTES) + "," +
                 FmtG(g_entry_px) + "," +
                 TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES) + "," +
                 FmtG(exit_px) + "," + outcome + "," + FmtG(pnl) + "," +
                 FmtG(g_tp_d) + "," + FmtG(g_sl_d) + "," +
                 FmtG(g_dir_open > 0 ? g_sig_pl : g_sig_ps);
   FileWrite(fh, line);
   FileClose(fh);
}

//+------------------------------------------------------------------+
double OnTester()
{
   PrintFormat("[summary] 评估K线%d根 信号%d个 成交%d笔", g_evals, g_signals, g_trades);
   return TesterStatistics(STAT_PROFIT);
}
//+------------------------------------------------------------------+
