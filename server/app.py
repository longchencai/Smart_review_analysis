# ============================================================
# 局域网演示服务：把「蒸馏+量化学生模型」和前端页面一起托管
#
# 用途：让同一个局域网里的同事直接用浏览器打开就能试
#       —— 不需要他们装 Python、下模型、配环境。
#
# 运行（在项目根目录下）：
#     D:\conda_envs\PYTHON_ML\python.exe -u server\app.py
#     # 默认监听 0.0.0.0:8000，启动后会打印局域网访问地址
#
# 常用参数：
#     --host 0.0.0.0    监听地址（默认 0.0.0.0，即局域网可达）
#     --port 8000       端口
#     --fp32            改用 fp32 权重（默认 int8，CPU 更快）
#     --reload          开发模式，改代码自动重启
#
# 接口：
#     GET  /              前端页面（frontend/index.html）
#     GET  /api/health    健康检查（前端用它自动判断该用真模型还是本地模拟）
#     GET  /api/info      模型元信息（参数量/体积/设备等）
#     POST /api/predict   预测，body: {"texts": ["...", ...]}  或  {"text": "..."}
#                         返回结构与 models/bert_distillation_quantization/predict.py
#                         完全一致（category/category_id/category_confidence/...）
#
# 设计说明：
#   · 模型在进程内是单例（predict.py 的 get_predictor 已做），启动时预热一次，
#     避免第一个请求要等模型加载。
#   · 模型文件缺失时服务仍能启动，接口返回 503 并给出可读原因 ——
#     这样前端页面不至于白屏，同事也能看懂缺什么。
#   · 量化模型只能跑 CPU（PyTorch int8 算子没有 CUDA 实现），这是硬约束。
# ============================================================

import argparse
import contextlib
import os
import socket
import sys
import time
import traceback

# ---- 把蒸馏模块目录加进 sys.path，才能 import 它的 predict / bert_config ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "bert_distillation_quantization")
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")
sys.path.insert(0, _MODEL_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# 缺省参数：--reload 模式下 uvicorn 会重新导入本文件而不执行 main()，
# 那时 _ARGS 还没被赋值。给个默认值，避免 AttributeError 让服务起不来。
_ARGS = argparse.Namespace(fp32=False, port=8000, host="0.0.0.0", reload=False)

# 进程内单例：避免每个请求都重新加载模型
_predictor = None
_load_error = None


class PredictRequest(BaseModel):
    texts: list[str] | None = None
    text: str | None = None
    top_k: int | None = None


def get_predictor_safe():
    """惰性取预测器；失败时把原因记下来，供接口返回可读信息。"""
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        from predict import get_predictor
        _predictor = get_predictor(quantized=not _ARGS.fp32, batch_size=64)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise RuntimeError(_load_error) from e
    return _predictor


def _warmup():
    """预热模型，顺便把加载结果打印出来。失败不阻断启动。"""
    t0 = time.time()
    try:
        p = get_predictor_safe()
        meta = p.info()
        print(f"  ✅ 模型已加载（{meta['dtype']} / {meta['device']} / "
              f"{meta['model_file_mb']} MB / 耗时 {time.time() - t0:.2f}s）")
        # 跑一条样例，确认前向真的通
        r = p.predict("热水器安装师傅很专业，出水温度稳定，洗澡很舒服")
        print(f"     自检预测: {r['category']} / {r['sentiment']}")
    except Exception as e:
        print(f"  ⚠️ 模型加载失败：{e}")
        print("     /api/predict 会返回 503，页面其余部分仍可正常打开。")


@contextlib.asynccontextmanager
async def lifespan(_app):
    """启动时预热模型。用 lifespan 而不是 @app.on_event —— 后者已被 FastAPI 废弃。"""
    _warmup()
    yield


app = FastAPI(title="电商评论智能分类 · 局域网演示", version="1.0", lifespan=lifespan)

# 允许跨源：同事如果把页面拷到自己电脑上直接打开，也能连过来
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ 前端页面
@app.get("/")
def index():
    page = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.exists(page):
        return JSONResponse(status_code=404, content={"error": f"找不到前端页面：{page}"})
    return FileResponse(page, media_type="text/html; charset=utf-8")


# ------------------------------------------------------------------ 健康检查
@app.get("/api/health")
def health():
    """前端加载时会先探测这个接口，决定走真实模型还是本地模拟。"""
    try:
        get_predictor_safe()
        return {"ok": True, "mode": "real"}
    except Exception as e:
        return {"ok": False, "mode": "error", "reason": str(e)}


# ------------------------------------------------------------------ 模型信息
@app.get("/api/info")
def info():
    try:
        return get_predictor_safe().info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ------------------------------------------------------------------ 预测
@app.post("/api/predict")
def predict(req: PredictRequest):
    texts = req.texts
    if not texts and req.text:
        texts = [req.text]
    if not texts:
        raise HTTPException(status_code=400, detail="请求体里需要 texts: [...] 或 text: '...'")

    try:
        predictor = get_predictor_safe()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型不可用：{e}")

    try:
        t0 = time.perf_counter()
        # with_truncation=True：额外做一次不截断的分词，把 token_length / truncated
        # 一并带回去 —— 页面要显示「多少 tokens / 是否已截断」。
        # 实测开销约 0.85 ms/条（7.2 → 8.05 ms/条，+12%），对演示场景可以接受。
        results = predictor.predict_batch(texts, top_k=(req.top_k or 3),
                                          with_truncation=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        # 把真实异常打回给调用方，方便同事定位（演示环境，可接受）
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败：{type(e).__name__}: {e}")

    # 单条时直接返回对象（和 predict.py 的输出结构一致），多条返回列表
    payload = results[0] if (req.text and not req.texts) else results
    headers = {"X-Inference-Ms": f"{elapsed_ms:.2f}",
               "X-Server-Total-Ms": f"{elapsed_ms:.2f}"}
    return JSONResponse(content=payload, headers=headers)


# ------------------------------------------------------------------ 静态资源
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


def local_ips():
    """列出本机在局域网里的可用 IPv4，供打印访问地址。"""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))      # 不会真的发包，只为让系统选出出口网卡
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips


def main():
    global _ARGS
    ap = argparse.ArgumentParser(description="局域网演示服务")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网可达）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fp32", action="store_true", help="用 fp32 权重（默认 int8）")
    ap.add_argument("--reload", action="store_true", help="开发模式自动重载")
    _ARGS = ap.parse_args()

    print("=" * 72)
    print("  电商评论智能分类 · 局域网演示服务")
    print("=" * 72)
    print(f"  项目根目录 : {_PROJECT_ROOT}")
    print(f"  模型目录   : {_MODEL_DIR}")
    print(f"  前端目录   : {_FRONTEND_DIR}")
    print(f"  权重精度   : {'fp32' if _ARGS.fp32 else 'int8（量化）'}")
    print()
    print("  本机访问   : " + f"http://127.0.0.1:{_ARGS.port}/")
    for ip in local_ips():
        print(f"  局域网访问 : http://{ip}:{_ARGS.port}/    ← 发给同事这个")
    print()
    print("  注意：如果同事打不开，通常是 Windows 防火墙拦了 —— 放行 Python 即可。")
    print("=" * 72)

    import uvicorn
    uvicorn.run("app:app" if _ARGS.reload else app,
                host=_ARGS.host, port=_ARGS.port, reload=_ARGS.reload)


if __name__ == "__main__":
    main()
