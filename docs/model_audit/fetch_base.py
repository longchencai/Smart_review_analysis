# -*- coding: utf-8 -*-
"""把 bert-base-chinese 的权重文件补到本地目录（只补权重，不动已存在的 config/tokenizer）。"""
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TARGET = r"D:\DEVELOP\Project\ai8_-project1\models\bert_distillation_quantization\bert-base-chinese"
print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
print(f"目标目录    = {TARGET}")
print(f"现有文件    = {sorted(os.listdir(TARGET))}")
print()

from huggingface_hub import hf_hub_download  # noqa: E402

ok = None
for fname in ("model.safetensors", "pytorch_model.bin"):
    try:
        t0 = time.time()
        print(f"下载 {fname} ...")
        p = hf_hub_download(repo_id="bert-base-chinese", filename=fname,
                            local_dir=TARGET, local_dir_use_symlinks=False)
        size = os.path.getsize(p)
        print(f"  OK -> {p}")
        print(f"  大小 {size:,} bytes = {size / 1024**2:.2f} MiB   耗时 {time.time() - t0:.1f}s")
        ok = p
        break
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:200]}")

print()
if ok:
    print("最终目录内容:")
    for f in sorted(os.listdir(TARGET)):
        fp = os.path.join(TARGET, f)
        print(f"  {os.path.getsize(fp):>12,}  {f}")
    print()
    print("=== 验证能否加载 ===")
    import transformers
    m = transformers.BertModel.from_pretrained(TARGET)
    print(f"  BertModel 加载成功，参数量 {sum(p.numel() for p in m.parameters()):,}")
else:
    print("两种权重文件都下载失败，需要改走代码补丁方案。")
