# ============================================================
# 步骤 2：蒸馏训练（SmallBERT 4L/384 学生）
#
# 三组递进实验（每组只比上一组多一个变量，便于归因）：
#   A = 基线 + warmup/余弦调度 + 更多轮数
#   B = A + 逐层 hidden-state 蒸馏（学生 1..4 层 ← 教师 3/6/9/12 层，384->768 投影）
#   C = B + 大类硬标签类别加权（针对占 macro 损失 42.5% 的「家用电器」）
#
# 损失：
#   loss = alpha*T^2*KL(教师‖学生) + (1-alpha)*CE(真实标签) + hidden_weight*MSE
#   · KL 方向：F.kl_div(input=学生, target=教师, log_target=True) 展开即 KL(教师‖学生)
#   · T^2 只乘软标签项（硬标签未做温度缩放）
#   · 投影层只在训练期存在，不进学生 state_dict，推理时丢弃
#
# 运行环境：GPU（dl-gpu）
#   python distill_train.py --exp A [--reset] [--teacher-ref] [--epochs 8]
# ============================================================

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from bert_config import Config
from dataloader_utils import build_all_dataloader
from distill_data import build_distill_dataloader
from report_utils import (append_block, count_parameters, evaluate, format_metrics_block,
                          reset_result_file, state_dict_size_mb)
from student_model import StudentBertMultiTask, describe_student

config = Config()

EXPERIMENTS = {
    'A': dict(schedule=True, hidden=False, class_weight=False,
              title='实验A：调度+更多轮数'),
    'B': dict(schedule=True, hidden=True, class_weight=False,
              title='实验B：A + 逐层hidden-state蒸馏'),
    'C': dict(schedule=True, hidden=True, class_weight=True,
              title='实验C：B + 类别加权'),
}


def apply_experiment(tag):
    spec = EXPERIMENTS[tag]
    config.distill_schedule = 'cosine' if spec['schedule'] else 'none'
    config.distill_use_hidden_loss = spec['hidden']
    config.distill_use_class_weight = spec['class_weight']
    config.student_model_tag = tag
    config.student_model_path = config.student_model_path.replace(
        'student_bert_4l384.pt', f'student_bert_4l384_exp{tag}.pt')
    return spec


def build_class_weights(labels, num_classes):
    """逆频率权重，归一化到均值 1，按上限截断后再归一化一次。"""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (1.0 / counts) ** config.distill_class_weight_power
    w = w / w.mean()
    w = np.minimum(w, config.distill_class_weight_max)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts


def build_scheduler(optimizer, steps_per_epoch):
    """OneCycleLR：前 pct_start 比例线性升到 max_lr，之后余弦退火。"""
    total = steps_per_epoch * config.distill_epochs
    if config.distill_schedule != 'cosine':
        return None, total, 0
    warmup = int(total * config.distill_warmup_ratio)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.distill_lr, total_steps=total,
        pct_start=config.distill_warmup_ratio, anneal_strategy='cos',
        div_factor=10.0, final_div_factor=100.0)
    return sched, total, warmup


def train_one_epoch(student, projs, loader, optimizer, scheduler, device, class_weight):
    student.train()
    T, alpha = config.distill_T, config.distill_alpha
    use_hidden = config.distill_use_hidden_loss
    acc = dict(loss=0.0, soft=0.0, hard=0.0, hidden=0.0)
    steps = 0

    for batch in loader:
        if use_hidden:
            enc, sent_y, cat_y, cat_soft, sent_soft, t_hidden = batch
            t_hidden = t_hidden.to(device)
        else:
            enc, sent_y, cat_y, cat_soft, sent_soft = batch

        enc = {k: v.to(device) for k, v in enc.items()}
        sent_y = sent_y.to(device)
        cat_y = cat_y.to(device)
        cat_soft = cat_soft.to(device)
        sent_soft = sent_soft.to(device)

        # 直接调内部 bert 以可选地拿 hidden_states；其后两层复现学生 forward 的完整计算
        out = student.bert(**enc, output_hidden_states=use_hidden)
        pooled = out['pooler_output']
        s_cat = student.cat_linear(pooled)
        s_sent = student.sent_linear(pooled)

        # ---- 软标签损失（两任务各一份 KL）----
        cat_soft_loss = F.kl_div(F.log_softmax(s_cat / T, -1), F.log_softmax(cat_soft / T, -1),
                                 reduction='batchmean', log_target=True)
        sent_soft_loss = F.kl_div(F.log_softmax(s_sent / T, -1), F.log_softmax(sent_soft / T, -1),
                                  reduction='batchmean', log_target=True)

        # ---- 硬标签损失（两任务各一份 CE；大类可选类别加权）----
        cat_hard_loss = F.cross_entropy(s_cat, cat_y, weight=class_weight)
        sent_hard_loss = F.cross_entropy(s_sent, sent_y)

        # ---- 组合：T^2 只补偿软标签项 ----
        soft = (cat_soft_loss + sent_soft_loss) / 2
        hard = (cat_hard_loss + sent_hard_loss) / 2
        loss = alpha * (T * T) * soft + (1 - alpha) * hard

        # ---- 逐层 hidden-state 蒸馏 ----
        hidden_loss = torch.zeros((), device=device)
        if use_hidden:
            for k, proj in enumerate(projs):
                s_cls = out.hidden_states[k + 1][:, 0, :]       # 学生第 k+1 层 [CLS] (B,384)
                hidden_loss = hidden_loss + F.mse_loss(proj(s_cls), t_hidden[:, k, :])
            hidden_loss = hidden_loss / len(projs)
            loss = loss + config.distill_hidden_weight * hidden_loss

        optimizer.zero_grad()
        loss.backward()
        if config.distill_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projs.parameters()), config.distill_grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        acc['loss'] += loss.item()
        acc['soft'] += soft.item()
        acc['hard'] += hard.item()
        acc['hidden'] += float(hidden_loss)
        steps += 1

    return {k: v / steps for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=list(EXPERIMENTS), required=True)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--reset', action='store_true', help='清空 eval_result.txt（只给第一个实验用）')
    ap.add_argument('--teacher-ref', action='store_true', help='额外评估教师作为参考基准')
    args = ap.parse_args()

    spec = apply_experiment(args.exp)
    if args.epochs:
        config.distill_epochs = args.epochs
    if args.reset:
        reset_result_file()

    print("=" * 70)
    print(f"步骤 2：蒸馏训练 —— {spec['title']}")
    print("=" * 70)
    device = config.device
    print(f"设备      : {device}")
    print(f"轮数/批次 : {config.distill_epochs} epochs x batch {config.distill_batch_size}")
    print(f"lr/调度   : lr={config.distill_lr}  schedule={config.distill_schedule}"
          f"  warmup_ratio={config.distill_warmup_ratio}")
    print(f"损失超参  : T={config.distill_T}  alpha={config.distill_alpha}")
    print(f"hidden蒸馏: {config.distill_use_hidden_loss}"
          + (f"  weight={config.distill_hidden_weight}  层映射={config.teacher_hidden_layer_map}"
             if config.distill_use_hidden_loss else ""))
    print(f"类别加权  : {config.distill_use_class_weight}")
    print(f"产出路径  : {config.student_model_path}")

    train_loader = build_distill_dataloader(shuffle=True)
    _, val_loader, _ = build_all_dataloader(batch_size=config.eval_batch_size)
    print(f"训练样本  : {len(train_loader.dataset)} 条 / {len(train_loader)} 批")
    print(f"验证样本  : {len(val_loader.dataset)} 条 / {len(val_loader)} 批")

    # ---- 类别权重 ----
    class_weight = None
    if config.distill_use_class_weight:
        class_weight, counts = build_class_weights(
            train_loader.dataset.cat_labels, config.cat_class_num)
        class_weight = class_weight.to(device)
        print(f"\n类别权重（逆频率^{config.distill_class_weight_power}，均值归一，上限 "
              f"{config.distill_class_weight_max:.0f}）:")
        for i in range(config.cat_class_num):
            print(f"    {config.id2cat[i]:<8} 样本 {int(counts[i]):>6}  权重 {class_weight[i].item():.3f}")

    # ---- 模型与投影层 ----
    student = describe_student()
    student.to(device)
    student_params = count_parameters(student)

    projs = torch.nn.ModuleList()
    if config.distill_use_hidden_loss:
        t_dim = config.bert_config.hidden_size
        projs = torch.nn.ModuleList([
            torch.nn.Linear(config.student_hidden, t_dim)
            for _ in config.teacher_hidden_layer_map]).to(device)
        proj_params = sum(p.numel() for p in projs.parameters())
        print(f"\n投影层    : {len(projs)} x Linear({config.student_hidden}->{t_dim})"
              f" = {proj_params:,} 参数（仅训练期存在）")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projs.parameters()),
        lr=config.distill_lr, betas=(0.9, 0.999), weight_decay=config.distill_weight_decay)
    scheduler, total_steps, warmup = build_scheduler(optimizer, len(train_loader))
    print(f"总步数    : {total_steps}（warmup {warmup} 步，OneCycle 余弦退火）")

    best_avg_f1, best_epoch = 0.0, 0
    history = []
    for epoch in range(1, config.distill_epochs + 1):
        t0 = time.time()
        stat = train_one_epoch(student, projs, train_loader, optimizer, scheduler, device, class_weight)
        print(f"\n[轮次 {epoch}/{config.distill_epochs}] 用时 {time.time() - t0:.1f}s  "
              f"总损失={stat['loss']:.4f}  软={stat['soft']:.4f}  硬={stat['hard']:.4f}"
              + (f"  hidden={stat['hidden']:.4f}" if config.distill_use_hidden_loss else ""))

        m = evaluate(student, val_loader, device=device)
        ca, cf = m['cat']['acc'], m['cat']['f1']
        sa, sf = m['sent']['acc'], m['sent']['f1']
        print(f"  验证集: 大类 acc={ca:.4f} f1={cf:.4f} | 情感 acc={sa:.4f} f1={sf:.4f} "
              f"| 综合 f1={m['avg_f1']:.4f}")
        history.append((epoch, cf, sf, m['avg_f1']))

        if m['avg_f1'] > best_avg_f1:
            best_avg_f1, best_epoch = m['avg_f1'], epoch
            torch.save(student.state_dict(), config.student_model_path)
            print(f"  ↑ 刷新最优综合 f1，已保存 -> {os.path.basename(config.student_model_path)}")

    # ---------------- 最终评估 ----------------
    print("\n" + "=" * 70)
    print(f"加载最优权重（第 {best_epoch} 轮）做最终验证集评估")
    print("=" * 70)
    student.load_state_dict(torch.load(config.student_model_path, map_location='cpu', weights_only=True))
    student.to(device)
    metrics = evaluate(student, val_loader, device=device, verbose=True)
    size = state_dict_size_mb(student)

    extra = [f"最优轮次: 第 {best_epoch} 轮 / 共 {config.distill_epochs} 轮",
             f"训练配置: lr={config.distill_lr}, schedule={config.distill_schedule}, "
             f"warmup={config.distill_warmup_ratio}, batch={config.distill_batch_size}",
             f"损失配置: T={config.distill_T}, alpha={config.distill_alpha}, "
             f"hidden蒸馏={config.distill_use_hidden_loss}"
             + (f"(weight={config.distill_hidden_weight})" if config.distill_use_hidden_loss else "")
             + f", 类别加权={config.distill_use_class_weight}",
             "逐轮综合f1: " + " -> ".join(f"e{e}:{a:.4f}" for e, _, _, a in history)]

    blocks = []
    if args.teacher_ref:
        print("\n评估教师模型作为参考基准 ...")
        from bert_classifier_model import MyBertMultiTaskClassifier
        teacher = MyBertMultiTaskClassifier()
        teacher.load_state_dict(torch.load(config.teacher_model_path, map_location='cpu', weights_only=True))
        teacher.to(device)
        tm = evaluate(teacher, val_loader, device=device)
        blocks.append(format_metrics_block(
            0, "教师模型 (BERT-base 12L/768, fp32)", "参考基准 —— 蒸馏要逼近的目标",
            tm, device, param_count=count_parameters(teacher),
            size_mb=state_dict_size_mb(teacher), model_path=config.teacher_model_path))
        del teacher
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    blocks.append(format_metrics_block(
        args.exp,
        f"{spec['title']}  （学生 SmallBERT {config.student_layers}L/{config.student_hidden}, fp32）",
        "蒸馏产出 —— 与【0】教师、【量化】段落同验证集、同评估代码",
        metrics, device, param_count=student_params, size_mb=size,
        model_path=config.student_model_path, extra=extra))

    append_block("\n".join(blocks))
    print(f"\n最优综合 f1 = {best_avg_f1:.4f}（第 {best_epoch} 轮）")
    print("步骤 2 完成")


if __name__ == '__main__':
    main()
