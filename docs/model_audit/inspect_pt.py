# -*- coding: utf-8 -*-
"""静态检查两个 .pt 权重文件：结构 / dtype / 数值分布 / 异常检测。
不依赖 transformers，只用 torch 直接读 state_dict。
v2: 支持 per-channel / per-channel-float-qparams 量化张量。"""
import hashlib
import os
import warnings

import torch

warnings.filterwarnings("ignore")

# 路径从脚本自身位置推算：本文件在 <项目根>/docs/model_audit/ 下，向上三级即项目根。
# 这样换机器、换 clone 目录都能直接跑，不需要改任何路径。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(_PROJECT_ROOT, "models", "bert_distillation_quantization", "model")
FILES = {
    "fp32": os.path.join(ROOT, "student_bert_4l384.pt"),
    "int8": os.path.join(ROOT, "student_bert_4l384_int8.pt"),
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def qinfo(t):
    qs = t.qscheme()
    if qs == torch.per_tensor_affine:
        return f"perTensor scale={t.q_scale():.6g} zp={int(t.q_zero_point())}", qs
    if qs == torch.per_channel_affine:
        sc = t.q_per_channel_scales()
        return (f"perChannelAffine axis={t.q_per_channel_axis()} n={sc.numel()} "
                f"scale[min={float(sc.min()):.4g},max={float(sc.max()):.4g}]"), qs
    if qs == torch.per_channel_affine_float_qparams:
        sc = t.q_per_channel_scales()
        zp = t.q_per_channel_zero_points()
        return (f"perChannelFloatQparams axis={t.q_per_channel_axis()} n={sc.numel()} "
                f"scale[min={float(sc.min()):.4g},max={float(sc.max()):.4g}] "
                f"zp[min={float(zp.min()):.4g},max={float(zp.max()):.4g}]"), qs
    return f"qscheme={qs}", qs


def tensor_stats(t):
    if t.is_quantized:
        ir = t.int_repr()
        vals = ir.to(torch.float32).flatten()
        info, qs = qinfo(t)
    else:
        vals = t.detach().to(torch.float32).flatten()
        info, qs = "", None
    n = vals.numel()
    d = dict(n=n, quant=t.is_quantized, info=info, qs=qs)
    if n == 0:
        return d
    d["nan"] = int(torch.isnan(vals).sum())
    d["inf"] = int(torch.isinf(vals).sum())
    fin = vals[torch.isfinite(vals)]
    if fin.numel() == 0:
        return d
    d.update(min=float(fin.min()), max=float(fin.max()), mean=float(fin.mean()),
             std=float(fin.std()) if fin.numel() > 1 else 0.0,
             zeros=int((fin == 0).sum()))
    return d


def analyse(tag, path):
    print("=" * 88)
    print(f"[{tag}]  {path}")
    print("=" * 88)
    if not os.path.exists(path):
        print("  !! 文件不存在")
        return None
    size = os.path.getsize(path)
    print(f"  文件大小   : {size:,} bytes = {size / 1024**2:.2f} MiB")
    print(f"  MD5        : {md5(path)}")

    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        print(f"  !! 顶层不是 dict：{type(sd)}")
        return None

    print(f"  条目数     : {len(sd)}")
    rows, non_tensor = [], []
    total_elem = 0
    q_cnt = f_cnt = 0
    theo = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            non_tensor.append((k, type(v).__name__, repr(v)[:70]))
            continue
        st = tensor_stats(v)
        total_elem += st["n"]
        if v.is_quantized:
            q_cnt += 1
            theo += v.int_repr().numel() + 8
        else:
            f_cnt += 1
            theo += v.numel() * v.element_size()
        rows.append((k, str(v.dtype), tuple(v.shape), st, v.is_quantized))

    print(f"  张量数     : {len(rows)}  (量化 {q_cnt} / 浮点 {f_cnt})")
    print(f"  非张量条目 : {len(non_tensor)}")
    for k, t, r in non_tensor:
        print(f"      {k}  ({t}) = {r}")
    print(f"  元素总数   : {total_elem:,}")
    print(f"  dtype 分布 : ")
    dc = {}
    for k, dt, sh, st, q in rows:
        dc[dt] = dc.get(dt, 0) + 1
    for dt, c in sorted(dc.items()):
        print(f"      {dt:<18} x {c}")

    print("\n  --- 按模块聚合存储量 (Top 20) ---")
    agg = {}
    for k, dt, sh, st, q in rows:
        parts = k.split(".")
        grp = ".".join(parts[:3]) if len(parts) > 3 else k
        b = (st["n"] + 8) if q else st["n"] * (4 if "float32" in dt else 8)
        agg[grp] = agg.get(grp, 0) + b
    for grp, b in sorted(agg.items(), key=lambda x: -x[1])[:20]:
        print(f"      {grp:<58} {b / 1024**2:>8.2f} MiB")

    print("\n  --- 浮点(未量化)条目 ---")
    any_f = False
    for k, dt, sh, st, q in rows:
        if q:
            continue
        any_f = True
        print(f"      {k:<56} {dt:<14} {str(sh):<14} n={st['n']:>8} "
              f"[{st.get('min', 0):.4g},{st.get('max', 0):.4g}] std={st.get('std', 0):.4g}")
    if not any_f:
        print("      (无)")

    print("\n  --- 量化条目（按前缀聚合，只列权重本身）---")
    shown = 0
    for k, dt, sh, st, q in rows:
        if not q:
            continue
        if not k.endswith("weight") and "_packed_params" not in k and "scale" not in k:
            continue
        if shown < 60:
            print(f"      {k:<56} {dt:<12} {str(sh):<14} n={st['n']:>8} "
                  f"int8[{st.get('min', 0):.0f},{st.get('max', 0):.0f}] zeros={st.get('zeros', 0):>7} {st['info']}")
            shown += 1
    if shown == 0:
        print("      (无)")

    print(f"\n  理论存储量 : {theo / 1024**2:.2f} MiB  vs 文件 {size / 1024**2:.2f} MiB "
          f"(差 {(size - theo) / 1024**2:+.2f} MiB)")

    print("\n  --- 异常检测 ---")
    anom = []
    for k, dt, sh, st, q in rows:
        n = st["n"]
        if n == 0:
            anom.append(f"{k}: 空张量")
        if st.get("nan"):
            anom.append(f"{k}: NaN x{st['nan']}")
        if st.get("inf"):
            anom.append(f"{k}: Inf x{st['inf']}")
        if st.get("zeros") == n and n > 1:
            anom.append(f"{k}: 全零")
        if k.endswith("LayerNorm.weight") and st.get("std", 1) < 1e-4:
            anom.append(f"{k}: LN gamma 退化 std={st['std']:.3g}")
        if q and "weight" in k and sh and len(sh) >= 2:
            zp_range = "zp" in st["info"]
            # 检测 int8 值域未用满
            if st.get("max", 0) - st.get("min", 0) < 60 and st["n"] > 1000:
                anom.append(f"{k}: int8 动态范围仅用 {st.get('max',0)-st.get('min',0):.0f}/255 档（量化粒度可能过细）")
    for a in anom[:40]:
        print(f"      !! {a}")
    if not anom:
        print("      未发现 NaN / Inf / 全零 / 退化")
    return dict(sd=sd, rows=rows, size=size, total_elem=total_elem)


def main():
    print(f"torch {torch.__version__}")
    res = {t: analyse(t, p) for t, p in FILES.items()}
    a, b = res.get("fp32"), res.get("int8")
    if a and b:
        ka, kb = set(a["sd"].keys()), set(b["sd"].keys())
        print("\n" + "=" * 88)
        print("[对比] fp32 vs int8")
        print("=" * 88)
        print(f"  键数: fp32={len(ka)}  int8={len(kb)}  共有={len(ka & kb)}")
        print(f"  fp32 独有 {len(ka - kb)} 个: {sorted(ka - kb)[:25]}")
        print(f"  int8 独有 {len(kb - ka)} 个(前25): {sorted(kb - ka)[:25]}")
        print(f"  元素总数: fp32={a['total_elem']:,}  int8={b['total_elem']:,}  "
              f"相等={a['total_elem'] == b['total_elem']}  差={b['total_elem'] - a['total_elem']:,}")
        print(f"  文件体积: {a['size'] / 1024**2:.2f} MiB -> {b['size'] / 1024**2:.2f} MiB  "
              f"= {b['size'] / a['size'] * 100:.1f}%  ({a['size'] / b['size']:.2f}x)")


if __name__ == "__main__":
    main()
