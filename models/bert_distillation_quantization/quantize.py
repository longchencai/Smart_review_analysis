# ============================================================
# 步骤 3：动态量化（PyTorch Dynamic Quantization, int8）
#
# 把蒸馏好的 SmallBERT 学生从 fp32 压成 int8，并在同一验证集上对比精度。
#
# 正确顺序（很关键，写反了必报 KeyError）：
#     建 fp32 学生 -> load fp32 权重 -> 评估 -> 就地 quantize_dynamic
#   而不是「先量化出结构，再往里灌 fp32 权重」——后者会报
#   KeyError: '...word_embeddings._packed_params.dtype'，因为量化模块
#   需要的是 _packed_params 系列键，和 fp32 的键完全对不上。
#   注意 quantize_dynamic 是「就地」替换子模块并返回同一个对象。
#   只有「加载已保存的量化权重」时才是「先建量化空壳再 load」。
#
# 三个必须遵守的约束（都踩过坑，实测确认）：
#   1. 量化过程必须在 CPU 上做。把 CUDA 模型直接丢给 quantize_dynamic
#      不会报错，但会产出一个 CPU/CUDA 都跑不了的废模型。
#   2. 量化结果只能跑 CPU。PyTorch 的 int8 算子（quantized::linear_dynamic
#      等）没有 CUDA kernel；对量化模型调用 .to('cuda') 也不报错，
#      但前向会抛 NotImplementedError。
#   3. Embedding 必须单独配 float_qparams_weight_only_qconfig。
#      学生词嵌入占其参数量的 52%，只量化 Linear 会丢掉一半以上压缩收益；
#      而把 nn.Embedding 用默认 qconfig 会直接 AssertionError。
#
# 运行环境：CPU（dl）
#   conda run -p C:\Users\29011\.conda\envs\dl python quantize.py
# ============================================================

import os
import time
import warnings

import torch
from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                   quantize_dynamic)

from bert_config import Config
from dataloader_utils import build_all_dataloader
from report_utils import (append_block, count_parameters, evaluate, format_compare_block,
                          format_metrics_block, state_dict_size_mb)
from student_model import StudentBertMultiTask

config = Config()


def build_qconfig_spec():
    if config.quantize_only_linear:
        return {torch.nn.Linear: default_dynamic_qconfig}
    return {
        torch.nn.Linear: default_dynamic_qconfig,
        torch.nn.Embedding: float_qparams_weight_only_qconfig,   # 不能省
    }


def quantize_model(model):
    """就地动态量化，返回同一个（已被替换子模块的）模型对象。"""
    model.eval()
    model.to('cpu')
    return quantize_dynamic(model, qconfig_spec=build_qconfig_spec(), dtype=torch.qint8)


def build_empty_quantized_student():
    """只造一个「量化后的空壳」，用于加载已保存的量化权重。"""
    return quantize_model(StudentBertMultiTask())


def timed_eval(model, loader, device, verbose=False):
    t0 = time.time()
    metrics = evaluate(model, loader, device=device, verbose=verbose)
    elapsed = time.time() - t0
    n = len(loader.dataset)
    metrics['_seconds'] = elapsed
    metrics['_ms_per_sample'] = elapsed / n * 1000
    return metrics


def main():
    # torch.ao.quantization 在 2.13 已标记 deprecated（提示迁移到 torchao），
    # 但当前仍是可用且最简的动态量化入口，这里屏蔽噪音警告、保留一个说明。
    warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'torch\.ao\.quantization')
    warnings.filterwarnings('ignore', message=r'.*quantize_per_tensor.*deprecated.*')

    print("=" * 68)
    print("步骤 3：动态量化（int8）")
    print("=" * 68)
    torch.backends.quantized.engine = config.quant_engine
    print(f"量化引擎  : {torch.backends.quantized.engine}")
    print(f"可用引擎  : {torch.backends.quantized.supported_engines}")
    print(f"量化范围  : {'仅 nn.Linear' if config.quantize_only_linear else 'nn.Linear + nn.Embedding(float_qparams)'}")
    print("说明      : torch.ao.quantization 已被标记 deprecated（官方建议迁移 torchao），"
          "但仍是当前最简可用的动态量化入口")

    if not os.path.exists(config.student_model_path):
        raise FileNotFoundError(
            f"找不到蒸馏好的学生权重 {config.student_model_path}，请先运行 distill_train.py")

    _, val_loader, _ = build_all_dataloader(batch_size=config.eval_batch_size)
    print(f"验证样本  : {len(val_loader.dataset)} 条 / {len(val_loader)} 批")

    # ---------------- [1] 量化前 fp32 基准（同为 CPU 评估，保证同设备可比）----------------
    print("\n[1/4] 加载 fp32 学生并评估（CPU）...")
    student = StudentBertMultiTask()
    missing, unexpected = student.load_state_dict(
        torch.load(config.student_model_path, map_location='cpu', weights_only=True), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"学生权重与结构不匹配！missing={list(missing)} unexpected={list(unexpected)}")
    student.eval().to('cpu')
    student_params = count_parameters(student)
    fp32_size = state_dict_size_mb(student)
    fp32_metrics = timed_eval(student, val_loader, 'cpu')
    print(f"  fp32  综合 f1={fp32_metrics['avg_f1']:.4f}  体积={fp32_size:.2f} MB  "
          f"{fp32_metrics['_ms_per_sample']:.2f} ms/条")

    # ---------------- [2] 就地量化 ----------------
    print("\n[2/4] 执行动态量化（就地替换子模块）...")
    t0 = time.time()
    q_student = quantize_model(student)
    q_student.eval().to('cpu')
    quant_seconds = time.time() - t0
    int8_size = state_dict_size_mb(q_student)
    print(f"  量化耗时 {quant_seconds:.2f}s   体积 {fp32_size:.2f} MB -> {int8_size:.2f} MB "
          f"({int8_size / fp32_size * 100:.1f}%)")

    # ---------------- [3] 量化后评估 ----------------
    print("\n[3/4] 量化模型在验证集上评估（CPU）...")
    int8_metrics = timed_eval(q_student, val_loader, 'cpu')
    print(f"  int8  综合 f1={int8_metrics['avg_f1']:.4f}  体积={int8_size:.2f} MB  "
          f"{int8_metrics['_ms_per_sample']:.2f} ms/条")

    # ---------------- [4] 保存 + 加载自检 ----------------
    print("\n[4/4] 保存量化模型并做加载自检 ...")
    torch.save(q_student.state_dict(), config.quantized_model_path)
    print(f"  已保存 -> {config.quantized_model_path}")

    # 加载自检：量化权重要先造出同样的量化空壳，再 load（与上面的 fp32 顺序相反）
    check = build_empty_quantized_student()
    missing, unexpected = check.load_state_dict(
        torch.load(config.quantized_model_path, map_location='cpu', weights_only=True), strict=False)
    check.eval().to('cpu')
    enc = config.bert_tokenizer(["这个键盘手感很好，打字很舒服", "酒店位置偏僻，服务很差"],
                                max_length=config.distill_max_len, padding='max_length',
                                truncation=True, return_tensors='pt')
    with torch.no_grad():
        c_out, s_out = check(enc)
    print(f"  加载自检: missing={list(missing)} unexpected={list(unexpected)}")
    print(f"  前向自检: cat={tuple(c_out.shape)} sent={tuple(s_out.shape)}  -> OK")
    if missing or unexpected:
        raise RuntimeError("量化权重加载自检失败：键不匹配")

    # ---------------- 写报告 ----------------
    blocks = []

    blocks.append(format_metrics_block(
        3, f"纯蒸馏学生 (SmallBERT {config.student_layers}L/{config.student_hidden}, fp32, CPU)",
        "量化前基准 —— 与【4】同设备、同代码，差值即为量化代价",
        fp32_metrics, 'cpu',
        param_count=student_params, size_mb=fp32_size,
        model_path=config.student_model_path,
        extra=[f"验证集推理耗时: {fp32_metrics['_seconds']:.1f}s "
               f"({fp32_metrics['_ms_per_sample']:.2f} ms/条, batch={config.eval_batch_size})"]))

    blocks.append(format_metrics_block(
        4, f"蒸馏 + 动态量化学生 (SmallBERT {config.student_layers}L/{config.student_hidden}, int8, CPU)",
        f"量化产出 —— 引擎={config.quant_engine}, "
        f"范围={'仅Linear' if config.quantize_only_linear else 'Linear + Embedding(float_qparams)'}",
        int8_metrics, 'cpu',
        param_count=student_params, size_mb=int8_size,
        model_path=config.quantized_model_path,
        extra=[f"验证集推理耗时: {int8_metrics['_seconds']:.1f}s "
               f"({int8_metrics['_ms_per_sample']:.2f} ms/条, batch={config.eval_batch_size})",
               f"体积压缩: {fp32_size:.2f} MB -> {int8_size:.2f} MB "
               f"({int8_size / fp32_size * 100:.1f}%, 即 {fp32_size / int8_size:.2f}x)",
               f"CPU 推理加速: {fp32_metrics['_ms_per_sample'] / int8_metrics['_ms_per_sample']:.2f}x",
               f"量化耗时: {quant_seconds:.2f}s（不含评估）"]))

    blocks.append(format_compare_block(
        "纯蒸馏学生 fp32 (CPU)", fp32_metrics,
        "蒸馏 + 动态量化 int8 (CPU)", int8_metrics))

    blocks.append(
        "  【量化约束备忘】\n"
        "    · 量化过程必须在 CPU 上执行（在 CUDA 模型上量化不报错，但产出废模型）\n"
        "    · 量化后的模型只能跑 CPU，PyTorch int8 算子没有 CUDA kernel\n"
        "    · 加载量化权重必须「先用同样 qconfig 量化出空壳，再 load_state_dict」；\n"
        "      加载 fp32 权重则是「先建 fp32 模型 load，再就地量化」，两者顺序相反\n"
        "    · 量化模型不可再训练（参数已冻结为 int8 整数）\n"
        "    · 参数量不变（仍是 %s），变的是存储精度与体积\n"
        % f"{student_params:,}")

    append_block("\n".join(blocks))
    print("\n步骤 3 完成")


if __name__ == '__main__':
    main()
