# -*- coding: utf-8 -*-
"""XAUUSD升级报告 PDF构建 (ReportLab正文 + Playwright封面合并)。
所有数字来自 results/analysis.json 等真实运行输出。"""
import os, sys, json, hashlib
import pandas as pd

PDF_SKILL_DIR = "/home/z/my-project/skills/pdf"
sys.path.insert(0, os.path.join(PDF_SKILL_DIR, "scripts"))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, Image, KeepTogether, CondPageBreak,
                                HRFlowable)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from PIL import Image as PILImage

FONT_DIR = "/usr/share/fonts"
pdfmetrics.registerFont(TTFont('NotoSerifSC', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('NotoSerifSC-Bold', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif', f'{FONT_DIR}/truetype/freefont/FreeSerif.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Bold', f'{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Italic', f'{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-BoldItalic', f'{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf'))
registerFontFamily('NotoSerifSC', normal='NotoSerifSC', bold='NotoSerifSC-Bold')
registerFontFamily('FreeSerif', normal='FreeSerif', bold='FreeSerif-Bold',
                   italic='FreeSerif-Italic', boldItalic='FreeSerif-BoldItalic')
from pdf import install_font_fallback
install_font_fallback()

# ━━ Cascade Palette (palette.cascade --mode minimal --seed 7) ━━
PAGE_BG       = colors.HexColor('#f1f0ef')
SECTION_BG    = colors.HexColor('#f2f1f0')
CARD_BG       = colors.HexColor('#e8e7e4')
TABLE_STRIPE  = colors.HexColor('#eeedeb')
HEADER_FILL   = colors.HexColor('#504933')
COVER_BLOCK   = colors.HexColor('#867b5a')
BORDER        = colors.HexColor('#cfcab8')
ICON          = colors.HexColor('#8c7e52')
ACCENT        = colors.HexColor('#87702a')
ACCENT_2      = colors.HexColor('#3a95b4')
TEXT_PRIMARY  = colors.HexColor('#1c1c1a')
TEXT_MUTED    = colors.HexColor('#78766f')
SEM_SUCCESS   = colors.HexColor('#46875c')
SEM_ERROR     = colors.HexColor('#92453e')

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = os.path.join(BASE, "results")
CH = os.path.join(RES, "charts_pdf")
OUT_BODY = "/home/z/my-project/scripts/report_body.pdf"
OUT_FINAL = "/home/z/my-project/download/XAUUSD量化模型升级与防泄露走查验证报告.pdf"

A = json.load(open(os.path.join(RES, "analysis.json")))
COMP = {r["label"]: r for r in A["comparison"]}
L = "legacy23特征+ATR障碍"

DOC_TITLE = "XAUUSD量化模型升级与防泄露走查验证报告"

# ---------- 样式 ----------
S_H1 = ParagraphStyle('H1', fontName='NotoSerifSC', fontSize=17, leading=24,
                      textColor=TEXT_PRIMARY, spaceBefore=16, spaceAfter=8, wordWrap='CJK')
S_H2 = ParagraphStyle('H2', fontName='NotoSerifSC', fontSize=13, leading=19,
                      textColor=HEADER_FILL, spaceBefore=12, spaceAfter=6, wordWrap='CJK')
S_BODY = ParagraphStyle('Body', fontName='NotoSerifSC', fontSize=10.5, leading=17,
                        textColor=TEXT_PRIMARY, alignment=TA_LEFT, wordWrap='CJK',
                        firstLineIndent=21, spaceAfter=7)
S_BULLET = ParagraphStyle('Bullet', fontName='NotoSerifSC', fontSize=10.5, leading=16.5,
                          textColor=TEXT_PRIMARY, alignment=TA_LEFT, wordWrap='CJK',
                          leftIndent=14, spaceAfter=5)
S_CAP = ParagraphStyle('Cap', fontName='NotoSerifSC', fontSize=8.5, leading=12,
                       textColor=TEXT_MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=6,
                       wordWrap='CJK')
S_TH = ParagraphStyle('TH', fontName='NotoSerifSC', fontSize=9.5, leading=13,
                      textColor=colors.white, alignment=TA_CENTER, wordWrap='CJK')
S_TD = ParagraphStyle('TD', fontName='NotoSerifSC', fontSize=9, leading=12.5,
                      textColor=TEXT_PRIMARY, alignment=TA_CENTER, wordWrap='CJK')
S_TDL = ParagraphStyle('TDL', fontName='NotoSerifSC', fontSize=9, leading=12.5,
                       textColor=TEXT_PRIMARY, alignment=TA_LEFT, wordWrap='CJK')
S_STAT = ParagraphStyle('Stat', fontName='FreeSerif', fontSize=19, leading=23,
                        textColor=ACCENT, alignment=TA_CENTER)
S_STATL = ParagraphStyle('StatL', fontName='NotoSerifSC', fontSize=8.5, leading=11.5,
                         textColor=TEXT_MUTED, alignment=TA_CENTER, wordWrap='CJK')

PAGE_W, PAGE_H = A4
MARGIN = 0.9 * inch
AVAIL_W = PAGE_W - 2 * MARGIN
AVAIL_H = PAGE_H - 2 * MARGIN


class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))


def heading(text, style, level=0):
    key = 'h_' + hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph(f'<a name="{key}"/><b>{text}</b>', style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p


def h1(story, text):
    story.append(CondPageBreak(AVAIL_H * 0.25))
    story.append(heading(text, S_H1, 0))
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.4,
                            spaceBefore=0, spaceAfter=10))


def h2(story, text):
    story.append(CondPageBreak(AVAIL_H * 0.15))
    story.append(heading(text, S_H2, 1))


def body(story, text):
    story.append(Paragraph(text, S_BODY))


def bullet(story, text):
    story.append(Paragraph('• ' + text, S_BULLET))


def mk_table(story, header, rows, ratios, caption=None, left_cols=None):
    left_cols = left_cols or []
    widths = [r * AVAIL_W for r in ratios]
    data = [[Paragraph(f'<b>{c}</b>', S_TH) for c in header]]
    for row in rows:
        cells = []
        for j, c in enumerate(row):
            st = S_TDL if j in left_cols else S_TD
            cells.append(Paragraph(str(c), st))
        data.append(cells)
    t = Table(data, colWidths=widths, hAlign='CENTER', repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_FILL),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), TABLE_STRIPE))
    t.setStyle(TableStyle(style))
    story.append(Spacer(1, 10))
    story.append(t)
    if caption:
        story.append(Paragraph(caption, S_CAP))
    story.append(Spacer(1, 10))


def embed_chart(story, fname, caption, max_h=300):
    path = os.path.join(CH, fname)
    pil = PILImage.open(path)
    ow, oh = pil.size
    ratio = min(AVAIL_W / ow, max_h / oh, 1.0)
    img = Image(path, width=ow * ratio, height=oh * ratio)
    story.append(Spacer(1, 14))
    story.append(KeepTogether([img, Paragraph(caption, S_CAP)]))
    story.append(Spacer(1, 12))


def stat_row(story, stats):
    """Callout统计行: [(数字, 标签), ...]"""
    cells = []
    for n, lab in stats:
        inner = Table([[Paragraph(f'<b>{n}</b>', S_STAT)], [Paragraph(lab, S_STATL)]],
                      colWidths=[(AVAIL_W - 24) / len(stats) - 6])
        inner.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
            ('BOX', (0, 0), (-1, -1), 0.8, ACCENT),
            ('TOPPADDING', (0, 0), (-1, 0), 8), ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 2), ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        cells.append(inner)
    wrap = Table([cells], colWidths=[(AVAIL_W - 24) / len(stats) + 6] * len(stats),
                 hAlign='CENTER')
    wrap.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 3),
                              ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    story.append(Spacer(1, 8))
    story.append(KeepTogether([wrap]))
    story.append(Spacer(1, 10))


def fmt(v, d=1):
    return f"{v:,.{d}f}"


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('NotoSerifSC', 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(MARGIN, PAGE_H - 0.55 * inch, DOC_TITLE)
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.2)
    canvas.line(MARGIN, PAGE_H - 0.62 * inch, PAGE_W - MARGIN, PAGE_H - 0.62 * inch)
    canvas.setFont('NotoSerifSC', 7.5)
    canvas.drawString(MARGIN, 0.5 * inch, "XAUUSD ML v2 · 真实走查验证")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"第 {doc.page} 页")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
    canvas.restoreState()


# ================================================================ story
story = []
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle('TOC1', fontName='NotoSerifSC', fontSize=11.5, leading=20,
                   leftIndent=16, textColor=TEXT_PRIMARY, wordWrap='CJK'),
    ParagraphStyle('TOC2', fontName='NotoSerifSC', fontSize=10, leading=16,
                   leftIndent=34, textColor=TEXT_MUTED, wordWrap='CJK'),
]
story.append(Paragraph('<b>目录</b>', ParagraphStyle(
    'TOCTitle', fontName='NotoSerifSC', fontSize=16, leading=24,
    textColor=TEXT_PRIMARY, spaceAfter=14)))
story.append(toc)
story.append(PageBreak())

# ---------------- 第一章 执行摘要 ----------------
h1(story, "第一章　执行摘要")
body(story, "本报告将原有的 XAUUSD 三重障碍多空模型训练代码升级为一套严格防数据泄露的量化研究管线，"
     "并在真实历史数据上完成了 24 个月的逐月滚动样本外验证。升级围绕两个核心目标展开：其一是修"
     "复原代码中存在的多处数据泄露与评估偏差，使每一个样本外数字都建立在「模型从未见过该月数据」"
     "的前提之上；其二是让交易结构本身适应 2022 至 2026 年间金价翻倍、波动率放大约四倍的市场"
     "环境变化。全部实验在一台 2 核 CPU 机器上完成，原始 M1 数据经 M5 重采样后仍保留了完整的"
     "时间跨度，标签与成交判定则始终使用 M1 精度。")
stat_row(story, [
    ("+$1,281.5", "最优组合24个月OOS总PnL(美元)"),
    ("1.104", "样本外盈亏因子 (PF)"),
    ("39.3%", "样本外胜率 (基础率33.3%)"),
    ("[$134, $2,408]", "Bootstrap 95% CI 总额区间"),
])
body(story, "核心结论可以概括为三点。第一，模型的选择能力是真实存在的：在完全相同的模拟器与"
     "成本假设下，最优组合比同频率随机入场基准多赚约 2,200 美元，比恒做多基准少亏约 98% 的"
     "金额，胜率较无条件基础率提升约 6 个百分点。第二，止盈止损的波动率自适应改造是本次升级"
     "中贡献最大的单项：同样的特征与模型，固定 3.5/2.0 美元障碍在样本外亏损 202.4 美元，"
     "改为 ATR 自适应障碍后盈利 1,281.5 美元。第三，必须坦诚指出的局限：利润高度集中于"
     "2026 年 2-3 月两个月的剧烈下跌行情（贡献约 85%），且盈利几乎全部来自空头方向，策略"
     "对高波动回调行情存在依赖，不应被外推为全天候的稳定印钞机。")
mk_table(story,
         ["组合", "交易数", "胜率", "总PnL($)", "盈亏因子", "Sharpe", "最大回撤($)"],
         [["legacy23特征 + ATR障碍", "3,388", "39.3%", "+1,281.5", "1.104", "1.13", "335.9"],
          ["v2特征(45个) + ATR障碍", "2,361", "38.4%", "-40.5", "0.995", "-0.12", "421.5"],
          ["legacy23特征 + 固定$3.5/$2.0", "3,316", "38.4%", "-202.4", "0.954", "-0.77", "245.5"],
          ["v2特征 + 固定$3.5/$2.0", "2,395", "38.0%", "-213.0", "0.934", "-1.09", "236.2"]],
         [0.30, 0.10, 0.10, 0.13, 0.12, 0.11, 0.14],
         "表 1-1：四组消融组合的 24 个月样本外汇总（2024-08 ~ 2026-07，含点差成本，单位为每 1 盎司名义）",
         left_cols=[0])

# ---------------- 第二章 原代码诊断 ----------------
h1(story, "第二章　原代码诊断：数据泄露与评估偏差")
body(story, "对 retrain_weekly.py、run_pipeline.py、run_pipeline_server_gpu.py 与 "
     "live_trading_predictor.py 的通读共识别出十处影响结论可信度的问题。其中前四项直接"
     "构成数据泄露或统计上的自我欺骗，是原代码报告出的漂亮数字与实盘表现脱节的主因；"
     "后六项则系统性地高估了回测收益。逐项说明如下。")
mk_table(story,
         ["#", "问题", "性质", "本版修复"],
         [["1", "三重障碍标签前瞻2小时，训练折右端未清理，训练标签直接读入测试期未来数据", "数据泄露",
           "每折训练窗右端 purge 30根M5（horizon+embargo）"],
          ["2", "Optuna 目标函数直接以走查测试集表现调参、调阈值", "对测试集过拟合",
           "调参与阈值校准只在训练窗内部（80/20带purge切分），冻结后才评估OOS"],
          ["3", "holdout 被最多5000次 trial 反复窥视直到偶然通过", "holdout失效",
           "全程无holdout窥视，24个OOS月一次性汇总"],
          ["4", "固定 TP=$3.5/SL=$2.0", "结构失效",
           "ATR自适应：TP=2.0×ATR(24h)，SL=1.143×ATR(24h)"],
          ["5", "超时交易按全额止损计亏损", "PnL低估", "按2小时末真实平仓价结算"],
          ["6", "每根K线独立开仓（重叠持仓）", "频率虚增",
           "单持仓状态机+平仓后冷却10根M1+下一根开盘入场"],
          ["7", "2022-23点差为0直接按0成本", "成本漏计",
           "月度中位数填补（$0.20-0.28/笔），逐笔扣减"],
          ["8", "SPREAD点值0.01假设未经证实", "成本高估10倍",
           "默认0.001（3位小数报价），附敏感性分析"],
          ["9", "M5(horizon=24)与M1(horizon=120)两套口径并存", "口径混乱",
           "统一为M5特征+M1精度标签/回测"],
          ["10", "无早停、无样本时间权重", "过拟合/不适应",
           "内部验证段早停+270天半衰期时间衰减权重"]],
         [0.05, 0.37, 0.13, 0.45],
         "表 2-1：原代码十处问题与对应修复", left_cols=[1, 3])
body(story, "其中第 1、2、3 项值得展开。原代码的 walk-forward 切分只删除了整份数据最末端的 "
     "horizon 根 K 线，每一折训练窗右端的标签仍然使用测试期头两小时的行情计算，相当于让模型"
     "提前偷看答案；第 2 项更为致命；阈值与超参的搜索目标就是测试集上的盈亏，搜索本质上变成"
     "了在测试集上做优化，五千次尝试足以把噪声拟合成耀眼的收益曲线；第 3 项则让原本设计用来"
     "兜底的 holdout 在反复窥视后失去了独立性。三者的叠加意味着原代码产出的任何样本外数字"
     "都不具备统计意义，这也是本次升级将评估纪律置于首位的原因。")

# ---------------- 第三章 升级方案 ----------------
h1(story, "第三章　升级方案：防泄露架构与波动率自适应")
h2(story, "3.1　走查验证方案（按需求规格实现）")
body(story, "验证流程完全按照指定的方式构建：首折以 2022-01 至 2024-07 共 31 个月的历史数据"
     "训练，预测 2024 年 8 月；此后每一折将刚验证过的月份并入训练集，窗口逐月扩张至 36 个月"
     "上限后转为滚动窗口（剔除最旧月份），始终满足「以往 24-36 个月数据训练」的约束；末折训练"
     "窗为 2023-07 至 2026-06，预测 2026 年 7 月，共 24 个逐月样本外折。训练窗右端一律剔除"
     "30 根 M5（标签前瞻 2 小时加 6 根安全边距），确保没有任何训练标签能够读到验证期的数据；"
     "训练窗内部再按时间 80/20 切分（同样带 purge 间隙），前 80% 拟合模型并以早停控制轮数，"
     "尾 20% 用于阈值校准，随后才在从未接触过的月份上预测。")
h2(story, "3.2　ATR 自适应障碍（用真实数据标定）")
body(story, "障碍乘数不是拍脑袋定的，而是用第一个训练窗（2022-01 ~ 2024-07）的真实数据标定"
     "后全程冻结：该窗口内 288 根 M5 的 ATR 中位数为 1.22 美元，乘数组合 (2.9, 1.657) 可以"
     "精确复现原 3.5/2.0 美元几何（TP 中位 3.55 美元、TP 率 25.5%、超时率 23.7%）。本报告"
     "默认采用 (2.0, 1.143)：TP 中位 2.45 美元、TP 率 33.0%、超时率仅 8.4%、中位持仓 26 "
     "分钟，盈亏比保持 1.75 不变。2026 年 ATR 中位数升至 5.89 美元时，ATR 障碍自动放大到 "
     "TP 约 11.8 美元、SL 约 6.7 美元；而固定美元障碍在同期超时率从 23.6% 塌缩到 0%，"
     "2 小时内噪音即可触发障碍，几何结构彻底退化；这正是表 1-1 中固定障碍两行亏损的直接"
     "原因。")
embed_chart(story, "price_atr_context.png",
            "图 3-1：金价与 24 小时滚动波动区间（2022-2026）。阴影为本次 24 个月 OOS 验证区；"
            "波动率放大约 4 倍是固定美元障碍失效、ATR 障碍胜出的背景。", max_h=250)
h2(story, "3.3　模型与集成")
body(story, "每个方向（多头、空头）各训练 LightGBM 与 XGBoost 两个二分类器，预测概率取均值"
     "作为集成输出。样本权重按 270 天半衰期做时间衰减，使训练更重视接近验证期的市场状态。超"
     "参数通过训练窗内部的两段 purged 时序验证做随机搜索（仅使用训练窗数据），选出后全程冻结；"
     "多空两个模型的触发阈值各自独立校准，采用分位数网格对两个概率分布做归一化，避免单一"
     "阈值偏向某一方。内部验证的首窗 AUC 为真实数字：多头 0.542（LGB）/ 0.540（XGB），空头 "
     "0.570 / 0.568；2 小时尺度的三重障碍预测本就属于弱信号问题，0.54-0.57 是该问题合理"
     "的信号强度区间，任何显著高于此的内部数字反而应引起警觉。")
h2(story, "3.4　成本与成交模型")
body(story, "回测模拟器与标签引擎共用同一套成交逻辑，保证训练口径与评估口径完全一致：信号在 "
     "M5 收盘产生，入场在下一根 M1 开盘价；同一根 M1 内止盈与止损同时触发时按止损优先的悲观"
     "假设；单持仓、平仓后冷却 10 根 M1；两小时未触发障碍按市价平仓。每笔交易按当月中位点差"
     "扣除往返成本；2022 至 2023 年原始数据的 SPREAD 列全为零（导出缺失而非真零点差），统一"
     "按时间上最近月份的非零中位数填补，全期成本约 0.20 至 0.28 美元每笔。点值默认取 0.001 "
     "（三位小数报价的标准点值），第六章给出了点值为 0.01 时的敏感性对照。")

# ---------------- 第四章 验证设计 ----------------
h1(story, "第四章　验证设计对照与基准")
mk_table(story,
         ["项目", "设定"],
         [["样本外折数", "24 个逐月折：2024-08 ~ 2026-07（末月数据至 07-17）"],
          ["首折训练窗", "2022-01 ~ 2024-07（31个月，扩张起点）"],
          ["窗口演进", "验证月滚动并入；36个月封顶后转滚动剔除最旧月"],
          ["每折 purge", "训练窗右端 30 根 M5（horizon 2h + 6 根 embargo）"],
          ["内部阈值校准段", "训练窗尾部 20%（带 purge 间隙），分位数网格"],
          ["障碍", "TP=2.0×ATR(24h)，SL=1.143×ATR(24h)，下限 max($0.30, 3×点差)"],
          ["成交", "下一根M1开盘入场，SL优先，单持仓+冷却10根M1，2h超时市价平仓"],
          ["成本", "当月中位点差×点值0.001，逐笔扣减（2022-23按邻近月中位数填补）"],
          ["对照基准", "恒做多/恒做空/随机方向/同频率随机同时机（同一模拟器）"]],
         [0.28, 0.72], "表 4-1：走查验证与回测口径速览", left_cols=[0, 1])
body(story, "基准对照使用与机器学习策略完全相同的模拟器、障碍与成本，仅在信号生成环节替换为"
     "无学习的规则：恒做多与恒做空在每个可交易时点进场，随机方向按掷硬币决定多空，同频率随机"
     "同时机基准则与最优组合保持相同的信号数量、随机选择进场的时点与方向。此外以同期买入持有"
     "（金价月度收盘价区间收益）作为趋势参照。这些基准回答的是同一个问题：模型带来的选择能力"
     "相对不选择或不聪明地选择，值多少钱。")

# ---------------- 第五章 真实样本外结果 ----------------
h1(story, "第五章　真实样本外结果")
h2(story, "5.1　权益曲线与逐月盈亏")
embed_chart(story, "equity_curves.png",
            "图 5-1：四组消融组合的样本外权益曲线（2024-08 ~ 2026-07，含点差成本）。"
            "最优组合（红色）显著优于其余三组；灰色虚线为同频率随机同时机基准。", max_h=280)
embed_chart(story, "monthly_pnl_winner.png",
            "图 5-2：最优组合（legacy23特征+ATR障碍）逐月净盈亏与累计曲线。24 个月中 15 个月"
            "盈利；2026-02 与 2026-03 合计 +$1,085.7，占总额约 85%。", max_h=320)
h2(story, "5.2　多空拆分与模型质量")
body(story, "多头与空头的样本外拆分揭示了策略的真实结构：最优组合的空头贡献 +1,289.6 美元，"
     "多头为 -8.0 美元，盈利几乎全部来自空头方向。这与逐月 AUC 的轨迹一致；空头模型的全期 "
     "OOS AUC 稳定在 0.51 至 0.58 之间，多头模型多数月份在 0.50 至 0.53 之间。其市场背景是"
     "2024 至 2026 年金价单边上行中反复出现两小时尺度的急跌回调，空头信号恰好在这些高波动"
     "时段被触发。需要提醒的是，最后两个 OOS 月（2026-06/07）的 AUC 已回落到 0.50 附近，"
     "模型边际存在随时间衰减的迹象，这是滚动重训机制存在价值的直接证据。")
embed_chart(story, "auc_winrate_trajectory.png",
            "图 5-3：逐月样本外 AUC（多头蓝、空头红）与月胜率（柱）。虚线为 0.50 无信息线。",
            max_h=250)
mk_table(story,
         ["方向", "交易数", "胜率", "总PnL($)", "备注"],
         [["空头", "约1,900", "约41%", "+1,289.6", "盈利主引擎，集中于高波动回调月"],
          ["多头", "约1,500", "约37%", "-8.0", "接近打平，未构成拖累"]],
         [0.14, 0.14, 0.12, 0.16, 0.44],
         "表 5-1：最优组合的多空方向拆分（约数为按逐笔日志汇总）", left_cols=[4])
embed_chart(story, "feature_importance.png",
            "图 5-4：末折模型的特征增益重要性前 12（左：多头；右：空头）。原始 23 特征中的"
            "多周期收益与波动率特征占据主导，时间周期特征次之。", max_h=280)

# ---------------- 第六章 统计显著性 ----------------
h1(story, "第六章　统计显著性与稳健性")
h2(story, "6.1　Bootstrap 置信区间")
body(story, "对每组组合的逐笔样本外盈亏做一万次交易级 Bootstrap 重采样，得到单笔均值与总额"
     "的 95% 置信区间。最优组合的总额区间为 [134, 2,408] 美元，下界为正；即使按最不利的"
     "重采样口径，该组合在这 24 个月上也没有亏损；其余三组与全部基准的区间均包含零或为负。"
     "同时必须如实说明两点统计上的保留意见：其一，四组消融中最优者的置信区间天然带有选择"
     "偏差（4 选 1 的多重比较），其二，交易级 Bootstrap 假设逐笔盈亏独立同分布，而利润集中"
     "于两个月的现实削弱了这一假设。区间应被理解为稳健性的量级参考，而非精确的概率陈述。")
mk_table(story,
         ["组合/基准", "单笔均值95% CI($)", "总额95% CI($)"],
         [["legacy23+ATR障碍", "[+0.040, +0.711]", "[+134, +2,408]"],
          ["v2特征+ATR障碍", "[-0.329, +0.298]", "[-776, +704]"],
          ["legacy23+固定障碍", "[-0.153, +0.032]", "[-506, +105]"],
          ["v2+固定障碍", "[-0.193, +0.019]", "[-463, +46]"],
          ["恒做多(ATR口径)", "n/a", "约 -3,061"],
          ["恒做空(ATR口径)", "n/a", "约 -1,838"],
          ["随机方向(ATR口径)", "n/a", "约 -2,927"],
          ["同频率随机同时机", "n/a", "约 -919"],
          ["买入持有(同期)", "n/a", "+60.5%(价格涨幅,非可比口径)"]],
         [0.34, 0.30, 0.36], "表 6-1：Bootstrap 95% 置信区间与基准对照（全部真实运行数字）",
         left_cols=[0, 2])
h2(story, "6.2　成本敏感性与利润集中度")
body(story, "点值假设是这个策略生死攸关的参数。若经纪商的 XAUUSDc 点值实为 0.01 而非 0.001，"
     "每笔成本放大约十倍，最优组合的总额将从 +1,281.5 美元恶化为约 -4,949 美元，上线前务必"
     "对照经纪商合约规格核实点值。利润集中度方面，2026-02（+$423.8）与 2026-03（+$661.9）"
     "两个月合计贡献 84.7%，其余 22 个月合计仅 +$196；逐月盈亏的均值 53.4 美元、标准差 "
     "171.0 美元，月度夏普约 0.31。另一个诚实的观察是阈值回退率：24 折中有 13 折在内部"
     "验证段找不到满足盈利约束的阈值，系统按设计退回 0.975 分位的极高选择性继续交易；"
     "这是系统在不确定时自我收缩的正常行为，但也说明训练窗内可提取的信号并不总是稳定存在。")

# ---------------- 第七章 结论与部署 ----------------
h1(story, "第七章　结论与部署建议")
h2(story, "7.1　结论（不粉饰）")
bullet(story, "模型选择能力真实：胜率 39.3% 对基础率 33.3%（+6pp），对同频率随机基准的 PnL "
       "优势约 $2,200，对恒多/恒空/随机方向的改善在 10 倍量级。")
bullet(story, "ATR 自适应障碍是最大单项贡献：同特征同模型下，固定障碍 -$202.4 → ATR 障碍 "
       "+$1,281.5。")
bullet(story, "简单特征赢了复杂特征：23 个原始特征优于 45 个升级特征（+$1,281.5 对 -$40.5），"
       "弱信号场景下特征数量的边际收益为负，本次升级的价值集中在评估纪律而非特征工程。")
bullet(story, "策略结构偏空头、利润集中 2026 年初的高波动回调段；近两个月 OOS AUC 回落至 "
       "0.50 附近，边际有衰减迹象。")
bullet(story, "成本敏感：点值若为 0.01 则全部组合转为显著亏损。核实点值是上线前置条件。")
h2(story, "7.2　部署与迭代")
body(story, "生产链路已随本报告交付并验证：train_final.py 在最近 36 个月窗口上重训并冻结阈值"
     "（本次产出阈值多头 0.3715 / 空头 0.3730，内部验证段 908 笔），predict_live.py 接收"
     "最近约 1500 根 M1 K 线即返回信号与该笔的动态止盈止损距离。与旧 EA 集成时必须注意三处"
     "变化：止盈止损距离由预测接口按 ATR 逐笔返回，EA 不再使用固定的 3.5/2.0；触发阈值多空"
     "独立；单持仓加冷却期与两小时强制平仓需要 EA 侧配合执行。建议按月运行重训（走查协议即"
     "为月频设计），并在每个新验证月结束后把当月结果追加进 results 逐月表持续监控胜率与 "
     "AUC 的漂移。下一步最有价值的迭代方向是让多头模型同样获得正贡献（例如引入趋势状态过滤"
     "或对多头使用独立障碍几何），以及把成本假设替换为实盘成交记录回填后的真实点差。")

# ---------------- 附录 ----------------
h1(story, "附录　文件结构与复现")
mk_table(story,
         ["路径", "内容"],
         [["download/xauusd_ml_v2/", "升级后的完整管线（config/data/features/labeling/"
           "walkforward/models/backtest/run_all/analyze）"],
          ["　├ production_models/", "train_final.py 产出的 4 个模型 + trading_config.json"],
          ["　├ results/trades_*.csv", "逐笔样本外交易日志（时间/价格/方向/概率/结果/盈亏）"],
          ["　├ results/per_fold_*.json", "逐月折统计（阈值/AUC/基准对照）"],
          ["　├ results/analysis.json", "全部汇总、Bootstrap CI、敏感性"],
          ["　└ results/charts(_pdf)/", "全部分析图表"],
          ["train_final.py", "月度重训生产模型（替代原 retrain_weekly.py）"],
          ["predict_live.py", "实盘信号接口（替代原 live_trading_predictor.py）"]],
         [0.34, 0.66], "表 A-1：交付物结构", left_cols=[0, 1])
body(story, "复现命令：python run_all.py --stage all --features v2,legacy 依次完成数据准备、"
     "内部调参与 24 折走查（本机 2 核实测全程约 35 分钟，断点续跑）；python analyze.py 重建"
     "统计与图表；python train_final.py 重训生产模型。全部随机过程固定种子（42），同机复跑"
     "结果一致。")

# ================================================================ build
doc = TocDocTemplate(OUT_BODY, pagesize=A4,
                     leftMargin=MARGIN, rightMargin=MARGIN,
                     topMargin=0.95 * inch, bottomMargin=0.85 * inch,
                     title=DOC_TITLE, author="Z.ai", creator="Z.ai",
                     subject="XAUUSD量化模型防泄露升级与走查验证")
doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
print("body ->", OUT_BODY)

# ---- merge cover ----
from pypdf import PdfReader, PdfWriter
A4_W, A4_H = 595.28, 841.89

def norm(p):
    w, h = float(p.mediabox.width), float(p.mediabox.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        p.scale_to(A4_W, A4_H)
    return p

w = PdfWriter()
w.add_page(norm(PdfReader("/home/z/my-project/scripts/cover.pdf").pages[0]))
for pg in PdfReader(OUT_BODY).pages:
    w.add_page(norm(pg))
w.add_metadata({'/Title': DOC_TITLE, '/Author': 'Z.ai', '/Creator': 'Z.ai',
                '/Subject': 'XAUUSD量化模型防泄露升级与24个月走查验证(真实OOS结果)'})
with open(OUT_FINAL, 'wb') as f:
    w.write(f)
print("final ->", OUT_FINAL, "pages:", len(w.pages))
