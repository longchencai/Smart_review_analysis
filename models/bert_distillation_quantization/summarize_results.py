# ============================================================
# 汇总脚本：解析 eval_result.txt 里的所有评估段落，生成横向对比表
#
# 用法（在 quantize.py 跑完之后执行）：
#   python summarize_results.py            # 只打印
#   python summarize_results.py --append   # 同时把对比段追加进 eval_result.txt
#
# 同时会把上一版（修正前的 4 轮基线）从 cache/eval_result_baseline_backup.txt
# 里解析出来，作为「基线」一行参与对比。
# ============================================================

import argparse
import os
import re

from bert_config import Config

config = Config()

RE_CAT = re.compile(r'商品大类\s*:\s*准确率=([\d.]+)\s+精确率=([\d.]+)\s+召回率=([\d.]+)\s+F1=([\d.]+)')
RE_SENT = re.compile(r'情感二分类\s*:\s*准确率=([\d.]+)\s+精确率=([\d.]+)\s+召回率=([\d.]+)\s+F1=([\d.]+)')
RE_AVG = re.compile(r'综合 F1\s*:\s*([\d.]+)')
RE_SIZE = re.compile(r'模型体积\s*:\s*([\d.]+) MB')
RE_PARAMS = re.compile(r'参数量\s*:\s*([\d,]+)')
RE_TITLE = re.compile(r'^【[^】]+】(.+)$', re.M)
RE_TIME = re.compile(r'\[写入时间 ([^\]]+)\]')


def parse_file(path):
    """把一个 eval_result.txt 拆成 [{tag, title, cat, sent, avg, size, params}]，按出现顺序。

    注意：必须按「行首的【」切块，不能用 str.split('【')——因为段落副标题里
    也会出现【0】、【量化】这类引用，会把块从中间截断（踩过一次）。
    """
    if not os.path.exists(path):
        return []
    text = open(path, encoding='utf-8').read()
    chunks = re.split(r'\n(?=【)', text)
    out = []
    for ch in chunks:
        ch = ch.lstrip('\n')
        if not ch.startswith('【') or '】' not in ch:
            continue
        tag = ch[1:].split('】')[0].strip()
        rest = ch.split('】', 1)[1]
        m_cat, m_sent, m_avg = RE_CAT.search(ch), RE_SENT.search(ch), RE_AVG.search(ch)
        if not (m_cat and m_sent and m_avg):
            continue
        title = rest.strip().splitlines()[0].strip() if rest.strip() else tag
        m_size, m_par = RE_SIZE.search(ch), RE_PARAMS.search(ch)
        out.append(dict(
            tag=tag, title=title,
            cat=dict(zip(('acc', 'pre', 'rec', 'f1'), map(float, m_cat.groups()))),
            sent=dict(zip(('acc', 'pre', 'rec', 'f1'), map(float, m_sent.groups()))),
            avg=float(m_avg.group(1)),
            size=float(m_size.group(1)) if m_size else None,
            params=int(m_par.group(1).replace(',', '')) if m_par else None,
        ))
    return out


def fmt_row(name, r, base=None, width=34):
    """一行表格；base 非空时附带相对差值。"""
    cur, sent, avg = r['cat']['f1'], r['sent']['f1'], r['avg']
    size = f"{r['size']:.2f}" if r['size'] is not None else '-'
    line = f"  {name:<{width}}{size:>10}{cur:>11.4f}{sent:>11.4f}{avg:>11.4f}"
    if base is not None:
        line += f"{avg - base['avg']:>+11.4f}"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--append', action='store_true', help='把对比段追加进 eval_result.txt')
    args = ap.parse_args()

    cur = parse_file(config.eval_result_path)
    base_file = os.path.join(os.path.dirname(config.teacher_soft_label_path),
                             'eval_result_baseline_backup.txt')
    base_rows = [r for r in parse_file(base_file) if '纯蒸馏学生' in r['title']]
    baseline = base_rows[0] if base_rows else None

    # 教师：取【0】
    teacher = next((r for r in cur if r['tag'] == '0'), None)

    # 实验组：标签为单个大写字母
    exps = [r for r in cur if re.fullmatch(r'[A-Z]', r['tag'])]
    # 量化组：标签形如 xxx-fp32 / xxx-int8
    quants = [r for r in cur if r['tag'].endswith(('-fp32', '-int8'))]

    W = 34
    L = "  " + "-" * (W + 44)
    lines = []
    lines.append("=" * 80)
    lines.append("【对比汇总】蒸馏实验 A/B/C + 量化")
    lines.append("-" * 80)
    header = f"  {'模型 / 配置':<{W}}{'体积MB':>10}{'大类F1':>11}{'情感F1':>11}{'综合F1':>11}{'Δvs上一步':>11}"
    lines.append(header)
    lines.append(L)

    prev = None
    if baseline:
        lines.append(fmt_row("基线 + 4轮恒定lr (修正前)", baseline))
        prev = baseline
    if teacher:
        lines.append(fmt_row("教师 BERT 12L/768 (上限)", teacher))
    lines.append(L)

    for r in exps:
        # 标题本身形如「实验A：调度+更多轮数  （学生 SmallBERT ...）」，取（之前的部分
        short = re.split(r'[（(]', r['title'])[0].strip()
        lines.append(fmt_row(short, r, base=prev))
        if prev is not None:
            prev = r
    if exps:
        lines.append(L)

    for r in quants:
        if r['tag'].endswith('-int8'):
            fp = next((q for q in quants if q['tag'] == r['tag'].replace('-int8', '-fp32')), None)
            lines.append(fmt_row(f"{r['tag']} (量化后)", r, base=fp))

    lines.append(L)
    if exps:
        best = max(exps, key=lambda r: r['avg'])
        lines.append(f"  最佳蒸馏实验         : 实验{best['tag']}  综合F1={best['avg']:.4f}")
        if teacher:
            lines.append(f"  对教师保留率         : {best['avg'] / teacher['avg'] * 100:.2f}%"
                         f"  (教师 {teacher['avg']:.4f})")
        if baseline:
            lines.append(f"  相对修正前基线提升   : {best['avg'] - baseline['avg']:+.4f}"
                         f"  ({baseline['avg']:.4f} -> {best['avg']:.4f})")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)

    if args.append:
        with open(config.eval_result_path, 'a', encoding='utf-8') as f:
            f.write("\n" + text + "\n")
        print(f"\n已追加到 {config.eval_result_path}")


if __name__ == '__main__':
    main()
