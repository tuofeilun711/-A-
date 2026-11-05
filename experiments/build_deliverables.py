"""Build the technical report PDF and defense deck from saved results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor as PptRGB
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PInches
from pptx.util import Pt as PPt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
DELIVERABLES = ROOT / 'deliverables'
NAVY = '#16324F'
TEAL = '#2A9D8F'
ORANGE = '#E76F51'
GRAY = '#6B7280'


def json_file(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def csv_file(name):
    with (RESULTS / name).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


SUMMARY = json_file('summary.json')
DIAGNOSTICS = json_file('model_diagnostics.json')
RELIABILITY = json_file('reliability_summary.json')
TRANSITION = json_file('transition_summary.json')
COMMON = csv_file('common_reference.csv')
VALIDATION = csv_file('validation_reference.csv')


def font_names():
    regular = Path(r'C:\Windows\Fonts\Deng.ttf')
    bold = Path(r'C:\Windows\Fonts\Dengb.ttf')
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont('ReportFont', str(regular)))
        pdfmetrics.registerFont(TTFont('ReportFontBold', str(bold)))
        return 'ReportFont', 'ReportFontBold'
    return 'Helvetica', 'Helvetica-Bold'


def pdf_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign='CENTER')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'ReportFontBold'),
        ('FONTNAME', (0, 1), (-1, -1), 'ReportFont'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#D7E0E4')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F6F8F9')]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf():
    regular, bold = font_names()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        'TitleCN',
        parent=styles['Title'],
        fontName=bold,
        fontSize=27,
        leading=36,
        textColor=colors.HexColor(NAVY),
        alignment=TA_CENTER,
    )
    subtitle = ParagraphStyle(
        'SubCN',
        parent=styles['Normal'],
        fontName=regular,
        fontSize=13,
        leading=20,
        textColor=colors.HexColor(GRAY),
        alignment=TA_CENTER,
    )
    h1 = ParagraphStyle(
        'H1CN',
        parent=styles['Heading1'],
        fontName=bold,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor(NAVY),
        spaceBefore=8,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        'BodyCN',
        parent=styles['BodyText'],
        fontName=regular,
        fontSize=9.7,
        leading=17,
        alignment=TA_JUSTIFY,
        wordWrap='CJK',
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        'CapCN',
        parent=body,
        fontSize=8,
        textColor=colors.HexColor(GRAY),
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    output = DELIVERABLES / 'A题_POD-AR_技术报告.pdf'
    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=44,
    )
    frame = Frame(48, 44, A4[0] - 96, A4[1] - 92, id='body')

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(48, 24, '题目 A · POD-AR 流场短时预报')
        canvas.drawRightString(A4[0] - 48, 24, str(document.page))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id='body', frames=[frame], onPage=footer)])
    story = [
        Spacer(1, 2.2 * cm),
        Paragraph('让 AI 预测下一阵风', title),
        Paragraph('基于 POD-AR 的二维矩形断面绕流短时预报', subtitle),
        Spacer(1, 1.0 * cm),
        pdf_table([
            ['验证集 RMSE', '测试集 RMSE', '相对持续性基线', 'POD 模态'],
            [
                f"{SUMMARY['validation_rmse']:.4f}",
                f"{SUMMARY['test_rmse']:.4f}",
                f"{SUMMARY['test_improvement_percent']:.1f}%",
                str(DIAGNOSTICS['n_modes']),
            ],
        ], widths=[4 * cm, 4 * cm, 4 * cm, 4 * cm]),
        Spacer(1, 0.8 * cm),
        Image(str(FIGURES / 'window_rmse.png'), width=16.5 * cm, height=7.1 * cm),
        Paragraph('图 1  validation 与冻结 test 的固定窗口误差', caption),
    ]
    story += [PageBreak()]
    story += [
        Paragraph('1 问题与数据边界', h1),
        Paragraph(
            '题面要求根据预测起点及之前的历史速度场，预测随后 20 帧，并在 test_a2 与 test_a4 '
            '的 99、199、299、379 四个固定窗口上评价。误差只在 mask=1 的 18,282 个流体网格上统计。',
            body,
        ),
        pdf_table([
            ['角色', '用途', '本实现'],
            ['train', '统计量与模型拟合', '归一化、POD、AR'],
            ['validation', '调参与稳定配置选择', '固定 4 窗口网格搜索'],
            ['test', '最终共同参照', '配置冻结后仅评价 8 窗口'],
            ['transition', '自主研究', '第三问压力测试'],
        ], widths=[3 * cm, 6 * cm, 7.5 * cm]),
        Spacer(1, 0.4 * cm),
        Paragraph(
            '原始数据文件 SHA-256 与官方值一致：b7253fa216b932a851d5a8d8213894f580cda8d03b120c6b90f08f3b715358ff。',
            body,
        ),
        Paragraph('2 POD-AR 模型', h1),
        Paragraph(
            f"模型在流体网格上拼接 w、u 两通道，得到 36,564 维状态；SVD 提取 "
            f"{DIAGNOSTICS['n_modes']} 个 POD 模态，解释 "
            f"{DIAGNOSTICS['pod_energy_percent']:.1f}% 的训练能量。随后在全部训练轨迹的模态系数上"
            f"联合构造 AR({DIAGNOSTICS['ar_order']}) 样本（不跨轨迹拼接窗口），以岭回归 "
            f"ridge={DIAGNOSTICS['ridge_alpha']:g} 稳定模态系数的单步演化。递推 20 步后重构完整流场。",
            body,
        ),
        Paragraph(
            f"AR 的单步训练 R²={DIAGNOSTICS['one_step_in_sample_r2']:.4f} 只说明一步系数拟合良好，"
            f"不能替代 20 步闭环误差。"
            '模型最终指标完全由固定窗口闭环递推得到。',
            body,
        ),
        Image(str(FIGURES / 'validation_sweep.png'), width=15.5 * cm, height=6.5 * cm),
        Paragraph('图 2 validation-only 超参数搜索', caption),
    ]
    story += [PageBreak()]
    story += [
        Paragraph('3 共同参照结果', h1),
        Paragraph(
            f"validation 四窗平均 RMSE 为 {SUMMARY['validation_rmse']:.6f}；"
            f"冻结后 test 八窗平均 RMSE 为 {SUMMARY['test_rmse']:.6f}，"
            f"持续性基线为 {SUMMARY['test_persistence_rmse']:.6f}，"
            f"相对改善 {SUMMARY['test_improvement_percent']:.1f}%。",
            body,
        ),
    ]
    rows = [['轨迹', '起点', '窗口平均 RMSE', '第 20 帧 RMSE']]
    for row in COMMON:
        rows.append([
            row['trajectory'],
            row['start'],
            f"{float(row['rmse_mean']):.4f}",
            f"{float(row['rmse_final']):.4f}",
        ])
    story += [pdf_table(rows, widths=[4 * cm, 3 * cm, 4 * cm, 4 * cm])]
    story += [PageBreak()]
    story += [
        Paragraph('4 可信性分析', h1),
        Paragraph(
            f"在 validation 上，模型误差从第 1 帧 {RELIABILITY['rmse_horizon_1']:.5f} "
            f"增至第 20 帧 {RELIABILITY['rmse_horizon_20']:.5f}；同期持续性基线平均为 "
            f"{RELIABILITY['persistence_rmse']:.5f}。",
            body,
        ),
        Paragraph(
            '为评估不确定度，保存训练 AR 单步残差，并对 20 个集成成员逐步重采样。'
            '随机种子固定为 20261026，结果可复现。集合离散度随误差增长而增大，'
            '可以作为相对风险信号，但不是经过覆盖率校准的概率区间。',
            body,
        ),
        Image(str(FIGURES / 'error_growth.png'), width=15.5 * cm, height=7.2 * cm),
        Paragraph('图 3 validation 误差增长与残差集成离散度', caption),
        Paragraph(
            f"当前配置谱半径为 {DIAGNOSTICS['spectral_radius']:.6f}，接近中性稳定。"
            '这支持 20 帧范围内的数值稳定，但不支持无限长时外推。',
            body,
        ),
    ]
    story += [PageBreak()]
    story += [
        Paragraph('5 第三问：transition 压力测试', h1),
        Paragraph(
            'transition 两条轨迹不作为训练、调参或共同参照测试，只用于跨工况压力实验。'
            '由于轨迹长度与工况不同，本节直接报告绝对误差，不把与 validation 的百分比差异解释为因果退化。',
            body,
        ),
    ]
    transition_rows = [['轨迹', '窗口数', '模型 RMSE', '持续性基线 RMSE']]
    for name, item in TRANSITION['transition'].items():
        transition_rows.append([
            name,
            str(item['windows']),
            f"{item['model_rmse_mean']:.5f}",
            f"{item['persistence_rmse_mean']:.5f}",
        ])
    story += [pdf_table(transition_rows, widths=[4 * cm, 3 * cm, 4 * cm, 4.5 * cm])]
    story += [
        Spacer(1, 0.3 * cm),
        Image(str(FIGURES / 'transition_rmse.png'), width=16 * cm, height=7 * cm),
        Paragraph('图 4 transition 滚动窗口误差（实线：POD-AR；虚线：持续性）', caption),
        Paragraph(
            '结果表明模型在两个 transition 子集上均保持短期预测优势，'
            '但不足以证明广义物理普适性。更长、更多工况的独立测试仍然必要。',
            body,
        ),
    ]
    story += [PageBreak()]
    story += [
        Paragraph('6 局限与复现', h1),
        Paragraph(
            '主要局限包括：只使用最后 10 帧历史；POD 截断损失约 18.5% 训练能量；'
            '谱半径接近 1；transition 样本较少；RMSE 不能单独证明涡结构完全正确。',
            body,
        ),
        Paragraph(
            '复现顺序：安装 requirements.txt，运行 experiments/q1_baseline.py 完成 validation 选参，'
            '再运行 main.py 生成 test 结果；随后运行 q2_reliability.py、q3_transfer.py、'
            'sweep_stable.py 和 make_figures_pil.py。',
            body,
        ),
        Paragraph(
            '仓库不包含原始 HDF5。AI 用于辅助代码检查、重构建议、图表和文档排版；'
            '所有指标由仓库脚本和冻结配置生成。',
            body,
        ),
    ]
    doc.build(story)
    return output


def ppt_text(slide, text, x, y, w, h, size=18, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(PInches(x), PInches(y), PInches(w), PInches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_top = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = 'Microsoft YaHei'
    run.font.size = PPt(size)
    run.font.bold = bold
    run.font.color.rgb = PptRGB.from_string(color.replace('#', ''))


def ppt_title(slide, title, subtitle=None):
    ppt_text(slide, title, 0.7, 0.35, 11.8, 0.5, size=25, bold=True)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, PInches(0.72), PInches(0.94), PInches(1.1), PInches(0.05)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = PptRGB.from_string(TEAL.replace('#', ''))
    bar.line.fill.background()
    if subtitle:
        ppt_text(slide, subtitle, 0.72, 1.02, 11.5, 0.35, size=12, color=GRAY)


def ppt_footer(slide, number):
    ppt_text(slide, '题目 A · POD-AR 流场短时预报', 0.72, 7.15, 5, 0.2, size=8, color=GRAY)
    ppt_text(slide, f'{number:02d}', 12.1, 7.15, 0.4, 0.2, size=8, color=GRAY, align=PP_ALIGN.RIGHT)


def ppt_metric(slide, x, y, w, label, value, accent=TEAL):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, PInches(x), PInches(y), PInches(w), PInches(1.0)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = PptRGB.from_string('F5F7F8')
    shape.line.color.rgb = PptRGB.from_string('DCE4E8')
    ppt_text(slide, label, x + 0.15, y + 0.12, w - 0.3, 0.22, size=10, color=GRAY)
    ppt_text(slide, value, x + 0.15, y + 0.40, w - 0.3, 0.35, size=22, color=accent, bold=True)


def build_pptx():
    presentation = Presentation()
    presentation.slide_width = PInches(13.333)
    presentation.slide_height = PInches(7.5)
    blank = presentation.slide_layouts[6]

    slide = presentation.slides.add_slide(blank)
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = PptRGB.from_string(NAVY.replace('#', ''))
    ppt_text(slide, '题目 A', 0.8, 0.6, 2, 0.35, size=14, color='#E9C46A', bold=True)
    ppt_text(slide, '让 AI 预测下一阵风', 0.8, 1.12, 10.5, 0.8, size=33, color='#FFFFFF', bold=True)
    ppt_text(
        slide,
        'POD 降维 + 轨迹内样本池化 AR 递推',
        0.82,
        2.0,
        8,
        0.45,
        size=17,
        color='#DDE7EC',
    )
    ppt_metric(slide, 0.82, 3.0, 2.7, 'Validation RMSE', f"{SUMMARY['validation_rmse']:.4f}", '#7DD3C7')
    ppt_metric(slide, 3.75, 3.0, 2.7, 'Sealed Test RMSE', f"{SUMMARY['test_rmse']:.4f}", '#F4B183')
    ppt_metric(slide, 6.68, 3.0, 2.7, 'Relative Improvement', f"{SUMMARY['test_improvement_percent']:.1f}%", '#F5DFA2')
    slide.shapes.add_picture(str(FIGURES / 'window_rmse.png'), PInches(0.6), PInches(4.55), width=PInches(8.1))

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '数据边界与防泄漏', 'test 只在配置冻结后评价')
    ppt_text(
        slide,
        'train\n归一化、POD、AR\n\nvalidation\n固定窗口调参与稳定性检查\n\n'
        'test\n仅最终 8 窗口共同参照\n\ntransition\n仅第三问压力测试',
        0.85, 1.5, 5.4, 3.8, size=18, color=NAVY, bold=True
    )
    ppt_metric(slide, 7.1, 1.45, 2.55, '流体网格', '18,282')
    ppt_metric(slide, 9.9, 1.45, 2.55, '真实状态维度', '36,564')
    ppt_metric(slide, 7.1, 2.8, 2.55, '输入历史', '10 帧')
    ppt_metric(slide, 9.9, 2.8, 2.55, '输出时域', '20 帧', ORANGE)
    ppt_text(
        slide,
        '起点之后的真值只参与 RMSE 计算。\n测试集不参与 n_modes、AR 阶数或 ridge 选择。',
        7.1, 4.55, 5.2, 1.2, size=16, color=GRAY
    )
    ppt_footer(slide, 2)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '模型：POD + AR', '每条轨迹单独构造 AR 样本')
    ppt_text(
        slide,
        '① mask 内拼接 (w,u) → 36,564 维\n\n'
        '② SVD → 50 个 POD 模态，81.5% 能量\n\n'
        '③ 模态系数上拟合 AR(3)\n\n'
        '④ ridge=3.0 抑制闭环递推不稳定\n\n'
        '⑤ 递推 20 步后重构完整流场',
        0.85, 1.55, 6.0, 4.5, size=19, color=NAVY
    )
    slide.shapes.add_picture(str(FIGURES / 'validation_sweep.png'), PInches(6.9), PInches(1.55), width=PInches(5.8))
    ppt_footer(slide, 3)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, 'Validation-only 选参', '先选出配置，再打开 test')
    ppt_metric(slide, 0.8, 1.5, 2.6, 'POD 模态', str(DIAGNOSTICS['n_modes']))
    ppt_metric(slide, 3.65, 1.5, 2.6, 'AR 阶数', str(DIAGNOSTICS['ar_order']))
    ppt_metric(slide, 6.5, 1.5, 2.6, 'Ridge', f"{DIAGNOSTICS['ridge_alpha']:g}", ORANGE)
    ppt_metric(slide, 9.35, 1.5, 3.0, '谱半径', f"{DIAGNOSTICS['spectral_radius']:.6f}")
    slide.shapes.add_picture(str(FIGURES / 'window_rmse.png'), PInches(0.75), PInches(3.0), width=PInches(11.7))
    ppt_footer(slide, 4)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '共同参照结果', '8 个固定窗口，全部纳入平均')
    ppt_metric(slide, 0.8, 1.45, 2.7, 'Test 八窗平均', f"{SUMMARY['test_rmse']:.6f}")
    ppt_metric(slide, 3.75, 1.45, 2.7, '持续性基线', f"{SUMMARY['test_persistence_rmse']:.6f}", GRAY)
    ppt_metric(slide, 6.7, 1.45, 2.7, '相对改善', f"{SUMMARY['test_improvement_percent']:.1f}%", ORANGE)
    ppt_metric(slide, 9.65, 1.45, 2.7, 'POD 能量', f"{DIAGNOSTICS['pod_energy_percent']:.1f}%", NAVY)
    rows = [['轨迹', '起点', '窗口均值', '第 20 帧']]
    for row in COMMON:
        rows.append([
            row['trajectory'], row['start'],
            f"{float(row['rmse_mean']):.4f}",
            f"{float(row['rmse_final']):.4f}",
        ])
    shape = slide.shapes.add_table(
        len(rows), 4, PInches(1.1), PInches(3.0), PInches(8.2), PInches(3.6)
    )
    table = shape.table
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            cell = table.cell(row_index, column)
            cell.text = value
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.name = 'Microsoft YaHei'
                    run.font.size = PPt(10)
    ppt_footer(slide, 5)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '误差增长与不确定度', 'validation-only，残差重采样集成')
    slide.shapes.add_picture(str(FIGURES / 'error_growth.png'), PInches(0.7), PInches(1.4), width=PInches(9.0))
    ppt_metric(slide, 9.95, 1.55, 2.6, '第 1 帧 RMSE', f"{RELIABILITY['rmse_horizon_1']:.5f}")
    ppt_metric(slide, 9.95, 2.9, 2.6, '第 20 帧 RMSE', f"{RELIABILITY['rmse_horizon_20']:.5f}", ORANGE)
    ppt_metric(
        slide,
        9.95,
        4.25,
        2.6,
        '集成成员',
        str(RELIABILITY.get('ensemble_members', '—')),
        NAVY,
    )
    ppt_footer(slide, 6)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '第三问：transition 压力测试', '只报告绝对误差，不做不可比百分比结论')
    slide.shapes.add_picture(str(FIGURES / 'transition_rmse.png'), PInches(0.7), PInches(1.4), width=PInches(9.2))
    ppt_metric(slide, 10.15, 1.55, 2.4, 'reverse_a0', f"{TRANSITION['transition']['reverse_a0']['model_rmse_mean']:.4f}")
    ppt_metric(slide, 10.15, 2.9, 2.4, 'reverse_a3', f"{TRANSITION['transition']['reverse_a3']['model_rmse_mean']:.4f}", ORANGE)
    ppt_text(
        slide,
        '结果支持有限跨工况迁移，\n不构成物理普适性证明。',
        10.1, 4.55, 2.6, 1.1, size=13, color=GRAY
    )
    ppt_footer(slide, 7)

    slide = presentation.slides.add_slide(blank)
    ppt_title(slide, '局限与结论')
    ppt_text(
        slide,
        '结论\n'
        f'• test 八窗 RMSE={SUMMARY["test_rmse"]:.4f}\n'
        f'• 相对持续性基线改善 {SUMMARY["test_improvement_percent"]:.1f}%\n'
        '• transition 上仍优于持续性\n\n'
        '局限\n'
        '• 只使用 10 帧历史\n'
        '• POD 忽略约 18.5% 训练能量\n'
        '• 谱半径接近 1，不支持无限长外推\n'
        '• transition 样本有限',
        0.9, 1.55, 11.6, 4.7, size=20, color=NAVY
    )
    ppt_footer(slide, 8)

    # This generated deck is an optional technical summary. The hand-edited
    # 11-slide defence deck is the submitted file and must not be overwritten.
    output = DELIVERABLES / 'A题_POD-AR_自动摘要.pptx'
    presentation.save(output)
    return output


def main():
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    pdf = build_pdf()
    pptx = build_pptx()
    print(pdf)
    print(pptx)


if __name__ == '__main__':
    main()
