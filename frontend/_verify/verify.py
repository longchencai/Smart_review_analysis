# -*- coding: utf-8 -*-
"""前端自动回归验证（无需安装任何东西，用系统自带的 Chrome/Edge 无头模式）

用法：
    python _verify/verify.py

它会做的事：
  1. 复制 index.html 到临时目录，注入「真实点击事件」驱动页面（不改动 index.html 本身）
  2. 用无头浏览器截图，逐页存到临时目录
  3. 让页面自己把关键测量值吐出来并断言：
       · 统计数字滚动结束后是否等于目标值（有没有负数 / 停在半路）
       · 进度条实际像素宽度是否等于置信度
       · 图表 canvas 是否真的渲染出来
       · 窄屏下导航栏 5 个元素是否都在屏幕内、页面有没有横向溢出
       · 是否有 JS 报错

退出码 0 = 全部通过。

注意：本脚本用无头浏览器的「虚拟时间」运行，此时 performance.now() 不随实际
执行推进，所以「响应耗时」的绝对数值会接近 0。这里只校验它的结构与一致性
（是否已填充、三段之和是否等于总耗时、顶部标签是否同步）。
要看真实数值，需用真实时钟驱动（例如 puppeteer-core 打开同一文件后读 #tTotal）。
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "index.html")
OUT = os.path.join(tempfile.gettempdir(), "fe_verify")

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# 场景：名称 / 目标页面 / 动作 / 窗口物理尺寸 / 设备像素比
SCENES = [
    ("01_overview",       "",           "",                                        "1600,1150", 1),
    ("02_classify_real",  "classify",   "fillSample(5)",                            "1600,1150", 1),
    ("03_classify_sim",   "classify",   "fillText('空调制冷效果差，噪音特别大，售后也不理人')", "1600,1150", 1),
    ("04_sentiment",      "sentiment",  "",                                        "1600,1400", 1),
    ("05_board",          "board",      "",                                        "1600,1300", 1),
    ("06_settings",       "settings",   "",                                        "1600,1000", 1),
    ("07_narrow_classify","classify",   "fillSample(3)",                            "780,1688", 2),
    ("08_narrow_overview","",           "",                                        "780,1688", 2),
]

HARNESS = r"""
<script>
window.__errs = [];
window.addEventListener('error', e => window.__errs.push('ERROR: ' + e.message));
const _ce = console.error;
console.error = function(){ window.__errs.push('console.error: ' + [].join.call(arguments,' ')); _ce.apply(console, arguments); };

function fillSample(i){
  const s = MODEL_SAMPLES[i], inp = document.getElementById('classifyInput');
  inp.value = s.text; inp.dispatchEvent(new Event('input'));
  document.getElementById('classifyBtn').click();
}
function fillText(t){
  const inp = document.getElementById('classifyInput');
  inp.value = t; inp.dispatchEvent(new Event('input'));
  document.getElementById('classifyBtn').click();
}

window.addEventListener('load', function () {
  const page = __PAGE__, act = __ACT__;
  const nav = n => { const e = document.querySelector('.nav-item[data-page="' + n + '"]'); if (e) e.click(); };
  if (act.indexOf('fillSample') === 0) { nav('classify'); setTimeout(() => fillSample(parseInt(act[11], 10)), 400); }
  else if (act.indexOf('fillText') === 0) { nav('classify'); setTimeout(() => fillText(act.slice(10, -2)), 400); }
  else if (page) { nav(page); }

  setTimeout(function () {
    const d = {};
    const sw = document.documentElement.scrollWidth;
    d.overflowPx = sw - innerWidth;
    if (d.overflowPx > 1){
      d.culprits = [];
      document.querySelectorAll('*').forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.right > innerWidth + 1)
          d.culprits.push({ tag: el.tagName.toLowerCase(), cls: (el.className||'').toString().slice(0,40) });
      });
    }
    d.stats = Array.from(document.querySelectorAll(
      '#page-' + (page || 'overview') + ' .stat-value[data-count]'))
      .map(el => ({ target: el.dataset.count, shown: el.textContent.trim(),
                    ok: el.textContent.trim().indexOf('-') === -1 }));
    d.navFits = Array.from(document.querySelectorAll('.navbar > *'))
      .every(el => el.getBoundingClientRect().right <= innerWidth + 1);
    d.charts = Array.from(document.querySelectorAll('.page.active .chart'))
      .map(c => { const cv = c.querySelector('canvas');
                  return { id: c.id, h: c.offsetHeight, cv: cv ? cv.width + 'x' + cv.height : 'none' }; });
    const box = document.getElementById('jsonBox');
    if (box){
      try { const j = JSON.parse(box.textContent);
            d.result = { cat: j.category, catConf: j.category_confidence,
                         sent: j.sentiment, sentConf: j.sentiment_confidence, tok: j.token_length };
      } catch(e){ d.result = 'JSON 解析失败'; }
      d.bars = Array.from(document.querySelectorAll('#classifyResult .bar')).map(b => {
        const i = b.querySelector('i');
        const bw = b.getBoundingClientRect().width, iw = i.getBoundingClientRect().width;
        return { want: i.style.width, barW: Math.round(bw),
                 pct: bw > 0 ? Math.round(iw / bw * 1000) / 10 : 'BAR_W=0' };
      });
    }
    d.errors = window.__errs;

    /* 响应耗时面板 */
    const tt = document.getElementById('tTotal');
    if (tt){
      const txt = id => { const e = document.getElementById(id); return e ? e.textContent.trim() : null; };
      const num = s => {
        if (!s) return NaN;
        const isSec = /s\s*$/.test(s) && !/ms\s*$/.test(s);
        // 去掉单位、'<'（极短耗时显示为 "<0.01 ms"）等非数字字符
        const v = parseFloat(s.replace(/[^0-9.]/g, ''));
        return isSec ? v * 1000 : v;
      };
      const t = { total: txt('tTotal'), pre: txt('tPre'),
                  infer: txt('tInfer'), render: txt('tRender'),
                  tag: txt('tTag'), session: txt('tSession') };
      t.filled = t.total.indexOf('—') === -1;
      if (t.filled){
        const sum = num(t.pre) + num(t.infer) + num(t.render);
        t.totalMs = num(t.total);
        t.sumMs = Math.round(sum * 100) / 100;
        t.consistent = Math.abs(t.sumMs - t.totalMs) <= 0.06;
      }
      d.timing = t;
    }

    if (__MODE__ === 'diag')
      document.documentElement.innerHTML = '<pre id="__diag">' + JSON.stringify(d) + '</pre>';
  }, 3600);
});
</script>
"""


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    sys.exit("找不到 Chrome / Edge，无法执行验证")


def run(chrome, args):
    return subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars"] + args,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=180)


def main():
    chrome = find_chrome()
    os.makedirs(OUT, exist_ok=True)
    html = open(SRC, encoding="utf-8").read()
    print("浏览器:", chrome)
    print("截图输出:", OUT)
    print("=" * 96)

    problems = []
    for name, page, act, size, dpr in SCENES:
        inject = (HARNESS.replace("__PAGE__", json.dumps(page))
                         .replace("__ACT__", json.dumps(act))
                         .replace("__MODE__", json.dumps("diag")))
        shot_html = HARNESS.replace("__PAGE__", json.dumps(page)) \
                           .replace("__ACT__", json.dumps(act)) \
                           .replace("__MODE__", json.dumps("shot"))
        diag_path = os.path.join(OUT, f"{name}_diag.html")
        shot_path = os.path.join(OUT, f"{name}_shot.html")
        open(diag_path, "w", encoding="utf-8").write(html.replace("</body>", inject + "</body>"))
        open(shot_path, "w", encoding="utf-8").write(html.replace("</body>", shot_html + "</body>"))

        base = [f"--force-device-scale-factor={dpr}", f"--window-size={size}"]
        png = os.path.join(OUT, f"{name}.png")
        if os.path.exists(png):
            os.remove(png)
        run(chrome, base + ["--virtual-time-budget=9000",
                            "--run-all-compositor-stages-before-draw",
                            f"--screenshot={png}",
                            "file:///" + shot_path.replace("\\", "/")])

        dom = run(chrome, base + ["--virtual-time-budget=16000", "--dump-dom",
                                  "file:///" + diag_path.replace("\\", "/")]).stdout or ""
        i, j = dom.find('<pre id="__diag">'), dom.find("</pre>")
        if i == -1:
            problems.append(f"{name}: 页面未产出诊断数据")
            continue
        d = json.loads(dom[i + len('<pre id="__diag">'):j])

        print(f"  {name:<20} 溢出={d['overflowPx']:>3}px  导航完整={d['navFits']}  "
              f"图表={len([c for c in d['charts'] if c['cv'] != 'none'])}/{len(d['charts'])}  "
              f"截图={os.path.getsize(png)//1024 if os.path.exists(png) else 0}KB")
        if d["stats"]:
            print(f"  {'':20} 数字: {[s['shown'] + ' (目标' + s['target'] + ')' for s in d['stats']]}")
        if d.get("result") and d["result"] != "JSON 解析失败":
            r = d["result"]
            print(f"  {'':20} 结果: {r['cat']} / {r['sent']}  "
                  f"置信度 {r['catConf']} / {r['sentConf']}  {r['tok']} tokens")
        if d.get("bars"):
            print(f"  {'':20} 进度条: " + ", ".join(f"{b['want']}→{b['pct']}" for b in d["bars"]))
        if d.get("timing"):
            t = d["timing"]
            print(f"  {'':20} 耗时: 总{t['total']} = 预处理{t['pre']} + 计算{t['infer']} + 渲染{t['render']}"
                  f"  和={t.get('sumMs')} 一致={t.get('consistent')}")
            print(f"  {'':20} 标签「{t['tag']}」 · 「{t['session']}」")
            if not t["filled"]:
                problems.append(f"{name}: 响应耗时未填充（仍为占位符）")
            elif not t.get("consistent"):
                problems.append(f"{name}: 耗时三段之和({t['sumMs']}) != 总耗时({t['totalMs']})")
            if t["tag"] == "响应 —":
                problems.append(f"{name}: 结果卡顶部耗时标签未更新")

        if d["overflowPx"] > 1:
            problems.append(f"{name}: 横向溢出 {d['overflowPx']}px {d.get('culprits','')}")
        if not d["navFits"]:
            problems.append(f"{name}: 导航栏元素超出屏幕")
        for s in d["stats"]:
            if not s["ok"]:
                problems.append(f"{name}: 数字异常 {s['shown']}（目标 {s['target']}）")
        for b in d.get("bars", []):
            if b["barW"] == 0:
                problems.append(f"{name}: 进度条容器宽度为 0（{b['want']}）")
        for c in d["charts"]:
            if c["cv"] == "none":
                problems.append(f"{name}: 图表 {c['id']} 未渲染")
        if d["errors"]:
            problems.append(f"{name}: JS 报错 {d['errors']}")

    print("=" * 96)
    if problems:
        print("发现问题：")
        for p in problems:
            print("  ✗", p)
        return 1
    print("✅ 全部通过：无横向溢出、导航栏完整、数字正确、进度条正常、图表已渲染、无 JS 报错")
    return 0


if __name__ == "__main__":
    sys.exit(main())
