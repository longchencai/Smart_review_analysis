# ============================================================
# predict.StudentPredictor 的测试
#
# 运行（用 GPU 环境：fp32 走 GPU、int8 走 CPU，两边都能测到）：
#   C:\Users\29011\.conda\envs\dl-gpu\python.exe -u test_predict.py
#
# 其中最硬的一条是「端到端一致性」：用预测器跑完整验证集，
# 看 macro-F1 能否复现 eval_result.txt 里记录的 0.9072(fp32) / 0.9074(int8)。
# 若能复现，说明预测器的预处理与离线评估完全一致。
# ============================================================

import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from bert_config import Config
from predict import StudentPredictor, architectural_param_count, get_predictor

config = Config()

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main():
    print("=" * 78)
    print("StudentPredictor 测试")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. 加载
    print("\n[1] 加载与元信息")
    fp32 = StudentPredictor(quantized=False).load()
    int8 = StudentPredictor(quantized=True).load()
    m32, m8 = fp32.info(), int8.info()
    check("int8 强制 CPU", m8['device'] == 'cpu', f"device={m8['device']}")
    check("fp32 设备跟随 config", m32['device'] == str(config.device), f"device={m32['device']}")
    check("参数量正确", m32['param_count'] == m8['param_count'] == 15560457,
          f"{m32['param_count']} / {m8['param_count']}")
    check("体积正确", abs(m8['model_file_mb'] - 15.12) < 0.1 and abs(m32['model_file_mb'] - 59.39) < 0.1,
          f"int8={m8['model_file_mb']}MB fp32={m32['model_file_mb']}MB")
    check("类别映射完整", len(m32['categories']) == 7 and len(m32['sentiments']) == 2)
    check("重复 load 幂等", fp32.load() is fp32)

    # ---------------------------------------------------------------- 2. 基本输出
    print("\n[2] 基本输出结构")
    r = int8.predict("这个键盘手感很好，打字很舒服，就是价格有点贵")
    need = {'text', 'category', 'category_id', 'category_confidence', 'sentiment', 'sentiment_id', 'sentiment_confidence'}
    check("返回字段齐全", need <= set(r), f"缺 {need - set(r)}" if not need <= set(r) else "")
    check("类别在合法集合内", r['category'] in m32['categories'], r['category'])
    check("情感在合法集合内", r['sentiment'] in m32['sentiments'], r['sentiment'])
    check("置信度在 (0,1]", 0 < r['category_confidence'] <= 1 and 0 < r['sentiment_confidence'] <= 1,
          f"{r['category_confidence']:.4f} / {r['sentiment_confidence']:.4f}")
    check("可 JSON 序列化", isinstance(__import__('json').dumps(r, ensure_ascii=False), str))
    print(f"        -> {r['category']} / {r['sentiment']}")

    rk = int8.predict("这个键盘手感很好", top_k=3)
    check("top_k 生效", len(rk['top_categories']) == 3)
    check("top_k 概率降序",
          all(rk['top_categories'][i]['probability'] >= rk['top_categories'][i + 1]['probability']
              for i in range(2)))

    # ---------------------------------------------------------------- 3. 确定性 & 批量一致性
    print("\n[3] 确定性与批量一致性")
    a = int8.predict("酒店位置偏僻，服务很差")
    b = int8.predict("酒店位置偏僻，服务很差")
    check("同输入同输出", a['category'] == b['category'] and a['sentiment'] == b['sentiment']
          and a['category_confidence'] == b['category_confidence'])

    texts = ["手机屏幕不错但电池太短", "酒店隔音差前台态度也不好", "书的内容精彩物流也快", "洗发水味道刺鼻"]
    bs = int8.predict_batch(texts, batch_size=1)
    b64 = int8.predict_batch(texts, batch_size=64)
    check("单条 vs 批量：预测标签一致",
          all(x['category'] == y['category'] and x['sentiment'] == y['sentiment'] for x, y in zip(bs, b64)))
    # 注意：动态量化会在「每次前向」按当前 batch 的激活 min/max 现算量化 scale，
    # 因此 batch 组成不同 → 置信度会有微小漂移。这是动态量化的固有行为，不是缺陷。
    # 对服务的含义：标签稳定；若业务要求置信度可复现，请固定 batch 组成（例如恒定 batch_size=1）。
    dmax = max(abs(x['category_confidence'] - y['category_confidence']) for x, y in zip(bs, b64))
    check("分块导致置信度漂移 < 0.01（动态量化固有行为）", dmax < 0.01, f"最大差 {dmax:.2e}")
    cpu32 = fp32.predict_batch(texts, batch_size=1)
    gpu32 = fp32.predict_batch(texts, batch_size=64)
    d32 = max(abs(x['category_confidence'] - y['category_confidence']) for x, y in zip(cpu32, gpu32))
    check("fp32 分块几乎无漂移", d32 < 1e-6, f"最大差 {d32:.2e}")

    # ---------------------------------------------------------------- 4. 边界情况
    print("\n[4] 边界情况")
    for bad, desc in [("", "空字符串"), ("   ", "纯空白"), ("\n\t ", "空白字符")]:
        try:
            int8.predict(bad)
            check(f"拒绝{desc}", False, "未抛异常")
        except ValueError:
            check(f"拒绝{desc}", True)
    try:
        int8.predict_batch([])
        check("拒绝空列表", False, "未抛异常")
    except ValueError:
        check("拒绝空列表", True)
    try:
        int8.predict_batch(["正常", "  "])
        check("批量里含空文本时报错", False, "未抛异常")
    except ValueError as e:
        check("批量里含空文本时报错", "1" in str(e), str(e)[:44])

    long_text = "这个手机很好用。" * 400
    rl = int8.predict(long_text, with_truncation=True)
    check("超长文本可预测", rl['category'] in m32['categories'])
    check("截断标记正确", rl['truncated'] is True, f"token_length={rl['token_length']}")
    rs = int8.predict("很好用", with_truncation=True)
    check("短文本未截断", rs['truncated'] is False, f"token_length={rs['token_length']}")

    # ---------------------------------------------------------------- 5. fp32 vs int8 一致率
    print("\n[5] fp32 vs int8 一致率（前 300 条验证集）")
    va = pd.read_csv(config.val_path, encoding=config.csv_encoding)
    sub = [str(t) for t in va['review'].head(300)]
    r32 = fp32.predict_batch(sub)
    r8 = int8.predict_batch(sub)
    agree_c = np.mean([a['category'] == b['category'] for a, b in zip(r32, r8)])
    agree_s = np.mean([a['sentiment'] == b['sentiment'] for a, b in zip(r32, r8)])
    check("大类一致率 >= 95%", agree_c >= 0.95, f"{agree_c * 100:.2f}%")
    check("情感一致率 >= 95%", agree_s >= 0.95, f"{agree_s * 100:.2f}%")

    # ---------------------------------------------------------------- 6. 端到端一致性
    print("\n[6] 端到端一致性：预测器跑完整验证集，复现 eval_result.txt 的指标")
    texts_all = [str(t) for t in va['review']]
    y_cat = np.array([config.cat2id[str(x)] for x in va['cat_l1']])
    y_sent = va['label'].to_numpy()

    expected = {
        'fp32': dict(cat_f1=0.8943, sent_f1=0.9202, avg=0.9072),
        'int8': dict(cat_f1=0.8943, sent_f1=0.9204, avg=0.9074),
    }
    for tag, predictor, dev in (('fp32', fp32, m32['device']), ('int8', int8, 'cpu')):
        t0 = time.time()
        res = predictor.predict_batch(texts_all, batch_size=64)
        el = time.time() - t0
        pc = np.array([config.cat2id[r['category']] for r in res])
        ps = np.array([config.sent2id[r['sentiment']] for r in res])
        cf = f1_score(y_cat, pc, average='macro', zero_division=0)
        sf = f1_score(y_sent, ps, average='macro', zero_division=0)
        exp = expected[tag]
        ok = (abs(cf - exp['cat_f1']) < 0.0005 and abs(sf - exp['sent_f1']) < 0.0005)
        check(f"{tag} 复现 大类F1={exp['cat_f1']} 情感F1={exp['sent_f1']}", ok,
              f"实测 {cf:.4f} / {sf:.4f}  综合 {(cf + sf) / 2:.4f}")
        print(f"        {tag}: {len(texts_all)} 条 / {el:.1f}s = {el / len(texts_all) * 1000:.2f} ms/条"
              f"  (设备 {dev}, batch=64)")

    # ---------------------------------------------------------------- 7. 单例
    print("\n[7] 单例复用")
    p1 = get_predictor(quantized=True)
    p2 = get_predictor(quantized=True)
    check("同参数返回同一实例", p1 is p2)
    check("单例已预热（load 耗时已记录）", p1.load_seconds is not None, f"{p1.load_seconds:.3f}s")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 78)
    print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
