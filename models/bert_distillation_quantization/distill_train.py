# ============================================================
# 步骤 2：蒸馏训练（SmallBERT 4L/384 学生）
#
# 损失：loss = alpha * T^2 * KL(教师‖学生) + (1 - alpha) * CE(真实标签)
#   · 软标签项：从缓存的教师 logits 读出，两个任务各算一份 KL
#   · 硬标签项：从 CSV 读真实标签，两个任务各算一份交叉熵
#   · T^2 只乘软标签项（硬标签没经过温度缩放，不能乘）
#   · KL 方向：F.kl_div(input=学生log_softmax, target=教师log_softmax, log_target=True)
#     展开即 KL(教师‖学生)，等价于 Hinton 的交叉熵形式，切勿调换两个参数
#
# 运行环境：GPU（dl-gpu）
#   conda run -p C:\Users\29011\.conda\envs\dl-gpu python distill_train.py
# ============================================================

import time

import torch
import torch.nn.functional as F

from bert_config import Config
from bert_classifier_model import MyBertMultiTaskClassifier
from dataloader_utils import build_all_dataloader
from distill_data import build_distill_dataloader
from report_utils import (append_block, count_parameters, evaluate, format_metrics_block,
                          reset_result_file, state_dict_size_mb)
from student_model import StudentBertMultiTask, describe_student

config = Config()


def train_one_epoch(student, loader, optimizer, device):
    student.train()
    T, alpha = config.distill_T, config.distill_alpha
    total_loss = total_soft = total_hard = 0.0
    steps = 0

    for enc, sent_y, cat_y, cat_soft, sent_soft in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        sent_y = sent_y.to(device)
        cat_y = cat_y.to(device)
        cat_soft = cat_soft.to(device)
        sent_soft = sent_soft.to(device)

        s_cat, s_sent = student(enc)

        # ---- 软标签损失（两个任务各一份 KL）----
        cat_soft_loss = F.kl_div(
            F.log_softmax(s_cat / T, dim=-1),
            F.log_softmax(cat_soft / T, dim=-1),
            reduction='batchmean', log_target=True)
        sent_soft_loss = F.kl_div(
            F.log_softmax(s_sent / T, dim=-1),
            F.log_softmax(sent_soft / T, dim=-1),
            reduction='batchmean', log_target=True)

        # ---- 硬标签损失（两个任务各一份 CE）----
        cat_hard_loss = F.cross_entropy(s_cat, cat_y)
        sent_hard_loss = F.cross_entropy(s_sent, sent_y)

        # ---- 组合：T^2 只补偿软标签项 ----
        soft = (cat_soft_loss + sent_soft_loss) / 2
        hard = (cat_hard_loss + sent_hard_loss) / 2
        loss = alpha * (T * T) * soft + (1 - alpha) * hard

        optimizer.zero_grad()
        loss.backward()
        if config.distill_grad_clip:
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.distill_grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_soft += soft.item()
        total_hard += hard.item()
        steps += 1

    return total_loss / steps, total_soft / steps, total_hard / steps


def main():
    print("=" * 68)
    print("步骤 2：蒸馏训练 SmallBERT 4L/384 学生")
    print("=" * 68)
    reset_result_file()
    device = config.device
    print(f"设备      : {device}")
    print(f"损失超参  : T={config.distill_T}  alpha={config.distill_alpha}  "
          f"epochs={config.distill_epochs}  batch={config.distill_batch_size}  lr={config.distill_lr}")
    print(f"教师软标签: {config.teacher_soft_label_path}")

    train_loader = build_distill_dataloader(shuffle=True)
    _, val_loader, _ = build_all_dataloader(batch_size=config.eval_batch_size)
    print(f"训练样本  : {len(train_loader.dataset)} 条 / {len(train_loader)} 批")
    print(f"验证样本  : {len(val_loader.dataset)} 条 / {len(val_loader)} 批")

    student = describe_student()
    student.to(device)
    student_params = count_parameters(student)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.distill_lr, betas=(0.9, 0.999),
        weight_decay=config.distill_weight_decay)

    best_avg_f1 = 0.0
    for epoch in range(1, config.distill_epochs + 1):
        t0 = time.time()
        loss, soft, hard = train_one_epoch(student, train_loader, optimizer, device)
        print(f"\n[轮次 {epoch}/{config.distill_epochs}] 用时 {time.time() - t0:.1f}s  "
              f"总损失={loss:.4f}  软={soft:.4f}  硬={hard:.4f}")

        metrics = evaluate(student, val_loader, device=device)
        ca, cf = metrics['cat']['acc'], metrics['cat']['f1']
        sa, sf = metrics['sent']['acc'], metrics['sent']['f1']
        avg = metrics['avg_f1']
        print(f"  验证集: 大类 acc={ca:.4f} f1={cf:.4f} | 情感 acc={sa:.4f} f1={sf:.4f} | 综合 f1={avg:.4f}")

        if avg > best_avg_f1:
            best_avg_f1 = avg
            torch.save(student.state_dict(), config.student_model_path)
            print(f"  ↑ 刷新最优综合 f1，已保存 -> {config.student_model_path}")

    # ---------------- 用最优权重做最终验证集评估 ----------------
    print("\n" + "=" * 68)
    print("加载最优学生权重，在验证集上做最终评估")
    print("=" * 68)
    student.load_state_dict(torch.load(config.student_model_path, map_location='cpu', weights_only=True))
    student.to(device)
    student_metrics = evaluate(student, val_loader, device=device, verbose=True)
    student_size = state_dict_size_mb(student)

    blocks = []

    # ---- 【1】教师参考基准：同一验证集、同一评估代码 ----
    print("\n评估教师模型作为参考基准 ...")
    teacher = MyBertMultiTaskClassifier()
    teacher.load_state_dict(torch.load(config.teacher_model_path, map_location='cpu', weights_only=True))
    teacher.to(device)
    teacher_metrics = evaluate(teacher, val_loader, device=device)
    teacher_params = count_parameters(teacher)
    teacher_size = state_dict_size_mb(teacher)
    blocks.append(format_metrics_block(
        1, "教师模型 (BERT-base 12L/768, fp32)", "参考基准 —— 蒸馏要逼近的目标",
        teacher_metrics, device,
        param_count=teacher_params, size_mb=teacher_size,
        model_path=config.teacher_model_path))
    del teacher
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- 【2】纯蒸馏学生 ----
    blocks.append(format_metrics_block(
        2, f"纯蒸馏学生 (SmallBERT {config.student_layers}L/{config.student_hidden}, fp32)",
        f"蒸馏产出 —— T={config.distill_T}, alpha={config.distill_alpha}, "
        f"{config.distill_epochs} epochs, 随机初始化 + 输出层蒸馏",
        student_metrics, device,
        param_count=student_params, size_mb=student_size,
        model_path=config.student_model_path,
        extra=[
            f"综合 f1 保留率 : {student_metrics['avg_f1'] / teacher_metrics['avg_f1'] * 100:.2f}%  (相对教师)",
            f"参数量压缩比   : {teacher_params / student_params:.2f}x",
            f"模型体积压缩比 : {teacher_size / student_size:.2f}x",
        ]))

    append_block("\n".join(blocks))
    print(f"\n最优综合 f1 = {best_avg_f1:.4f}")
    print("步骤 2 完成")


if __name__ == '__main__':
    main()
