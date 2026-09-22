# -*- coding: utf-8 -*-
"""验证：项目内的任何脚本能否在缺少基座权重的情况下启动。"""
import os
import sys
import traceback

# 路径从脚本自身位置推算：本文件在 <项目根>/docs/model_audit/ 下，向上三级即项目根。
# 这样换机器、换 clone 目录都能直接跑，不需要改任何路径。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.join(_PROJECT_ROOT, "models", "bert_distillation_quantization"))
sys.path.insert(0, os.getcwd())

print(f"cwd = {os.getcwd()}")
print(f"bert-base-chinese 目录内容: {sorted(os.listdir('bert-base-chinese'))}")
print("  -> 是否存在权重文件:",
      [f for f in os.listdir('bert-base-chinese')
       if f in ('pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'flax_model.msgpack')]
      or "【无】")
print("\n--- 尝试 import bert_config（这是所有脚本的第一行依赖）---")
try:
    import bert_config  # noqa
    print("OK: bert_config 导入成功")
except Exception:
    print("FAIL: bert_config 导入失败")
    traceback.print_exc()

print("\n--- 尝试直接 from_pretrained 加载基座 ---")
try:
    import transformers
    m = transformers.BertModel.from_pretrained("bert-base-chinese")
    print(f"OK: 加载成功 {type(m).__name__}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}")
    print(str(e)[:1400])
