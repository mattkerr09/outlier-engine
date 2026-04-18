"""
Outlier Desktop App — Backend v0.2 (Inference Mode)
Wires real inference via outlier_engine when available.
Falls back to stub mode when OUTLIER_SKIP_MODEL=1 is set.
"""
import os, json, signal, sys, time, asyncio, threading
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

APP_VERSION = "0.2.0"
SKIP_MODEL = os.environ.get("OUTLIER_SKIP_MODEL", "0") == "1"
MODEL_REF = "Outlier-Ai/Outlier-10B-V3.2"

DATA_DIR = Path.home() / ".outlier"
DATA_DIR.mkdir(exist_ok=True)
PROFILES_FILE = DATA_DIR / "profiles.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"

DEFAULT_PROFILES = {
    "default": {"name": "Default", "color": "#888888", "alpha": [0.0] * 224},
    "medical": {"name": "Medical", "color": "#ff6b6b", "alpha": [0.5] * 224},
    "coding": {"name": "Coding", "color": "#51cf66", "alpha": [0.4] * 224},
    "legal": {"name": "Legal", "color": "#ffd43b", "alpha": [0.6] * 224},
}
if not PROFILES_FILE.exists():
    PROFILES_FILE.write_text(json.dumps(DEFAULT_PROFILES, indent=2))

# ---------------------------------------------------------------------------
# Model state (global, loaded in background thread)
# ---------------------------------------------------------------------------
model_ready = False
loaded_model = None
model_load_error = None
_shutdown_event = threading.Event()


def _load_model_background():
    """Load the model in a background thread so the server starts immediately."""
    global model_ready, loaded_model, model_load_error
    try:
        from outlier_engine.loader import load_model
        loaded_model = load_model(MODEL_REF, paged=True)
        model_ready = True
        print(f"[desktop] Model {MODEL_REF} loaded successfully.")
    except Exception as exc:
        model_load_error = str(exc)
        print(f"[desktop] Model load FAILED: {exc}", file=sys.stderr)


def _unload_model():
    """Free model memory on shutdown."""
    global loaded_model, model_ready
    if loaded_model is not None:
        print("[desktop] Unloading model...")
        del loaded_model
        loaded_model = None
        model_ready = False
        try:
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Outlier Desktop", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    profile: str = "default"
    max_tokens: int = 512
    temperature: float = 0.7


class ProfileUpdate(BaseModel):
    profile: str
    alpha: list


@app.on_event("startup")
async def startup_event():
    if not SKIP_MODEL:
        t = threading.Thread(target=_load_model_background, daemon=True)
        t.start()
    else:
        print(f"[desktop] OUTLIER_SKIP_MODEL=1 — running in stub mode.")


@app.on_event("shutdown")
async def shutdown_event():
    _unload_model()


# Register SIGTERM handler for graceful shutdown outside uvicorn
def _sigterm_handler(signum, frame):
    _unload_model()
    _shutdown_event.set()
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")


@app.get("/api/version")
def version():
    mode = "inference" if (model_ready and not SKIP_MODEL) else "stub"
    return {"version": APP_VERSION, "mode": mode, "model": MODEL_REF}


@app.get("/api/profiles")
def get_profiles():
    return json.loads(PROFILES_FILE.read_text())


@app.post("/api/profile")
def update_profile(req: ProfileUpdate):
    profiles = json.loads(PROFILES_FILE.read_text())
    if req.profile not in profiles:
        return {"error": "unknown profile"}
    profiles[req.profile]["alpha"] = req.alpha
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2))
    return {"ok": True}


@app.get("/api/history")
def get_history():
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text().splitlines() if line.strip()]


@app.post("/api/clear-history")
def clear_history():
    HISTORY_FILE.write_text("")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Stub fallback
# ---------------------------------------------------------------------------
STUB_RESPONSES = {
    "default": "I'm Outlier, a ternary-quantized MoE language model running locally on your machine. In stub mode I return canned responses, but the full inference pipeline is wired up and ready to swap in once the engine is connected. The actual 10B V3.2 model scored 76.26% MMLU at 99.1% teacher retention.",
    "medical": "Speaking from a medical context: this is a stubbed response for testing the profile switcher. When connected to the real backend, the medical profile suppresses non-clinical experts and biases routing toward layers that learned medical content during distillation. The 224 alpha scalars are persisted to ~/.outlier/profiles.json.",
    "coding": "```python\n# Stub response in coding profile\ndef hello_outlier():\n    return 'I am running locally'\n```\n\nWhen the backend is wired up, the coding profile activates layer-2 expert-1 most strongly (alpha=0.86 from Day 7 Exp 1 results). Profile switching latency is 0.332ms.",
    "legal": "From a legal-context perspective: this is a stubbed response for UI testing. The legal profile in production would route through experts that learned legal corpus content. Note: Outlier is not a lawyer. Consult a licensed attorney for actual legal advice.",
}


async def _stream_stub(message: str, profile: str):
    """Stub streaming — word-by-word with simulated latency."""
    full_text = STUB_RESPONSES.get(profile, STUB_RESPONSES["default"])
    words = full_text.split(" ")
    for w in words:
        yield (w + " ").encode()
        await asyncio.sleep(0.04)


async def _stream_inference(message: str, req: ChatRequest):
    """Real inference streaming via outlier_engine.generate.stream_generate."""
    from outlier_engine.generate import stream_generate

    token_count = 0
    t0 = time.perf_counter()
    for token_text in stream_generate(
        loaded_model,
        message,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    ):
        token_count += 1
        yield token_text.encode("utf-8")

    # After all tokens are emitted, send a metadata footer the frontend can parse
    elapsed = time.perf_counter() - t0
    tps = token_count / elapsed if elapsed > 0 else 0.0
    # Delimiter so frontend can separate metadata from content
    yield f"\n\x00META:{json.dumps({'tok_count': token_count, 'elapsed_s': round(elapsed, 3), 'tok_per_s': round(tps, 1)})}".encode("utf-8")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    use_inference = model_ready and not SKIP_MODEL and loaded_model is not None

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "profile": req.profile,
        "user": req.message,
        "stub": not use_inference,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if use_inference:
        return StreamingResponse(
            _stream_inference(req.message, req),
            media_type="text/plain",
        )
    else:
        return StreamingResponse(
            _stream_stub(req.message, req.profile),
            media_type="text/plain",
        )


@app.get("/api/health")
def health():
    if SKIP_MODEL:
        mode = "stub"
        loaded = False
    else:
        mode = "inference" if model_ready else "loading"
        loaded = model_ready

    resp = {
        "status": "ok",
        "mode": mode,
        "model_loaded": loaded,
        "model": MODEL_REF,
        "version": APP_VERSION,
    }
    if model_load_error:
        resp["error"] = model_load_error
    return resp


# ---------------------------------------------------------------------------
# HF model tier management (Day 17 mega sprint: Track 2.6)
# ---------------------------------------------------------------------------
MODEL_TIERS = {
    "nano": {"repo": "Outlier-Ai/Outlier-Nano-1.7B-MLX-4bit", "size_gb": 0.96, "ram_req_gb": 2, "display": "Outlier Nano (1.7B)"},
    "lite": {"repo": "Outlier-Ai/Outlier-Lite-7B-MLX-4bit", "size_gb": 4.36, "ram_req_gb": 6, "display": "Outlier Lite (7B)"},
    "compact": {"repo": "Outlier-Ai/Outlier-Compact-14B-MLX-4bit", "size_gb": 8.42, "ram_req_gb": 12, "display": "Outlier Compact (14B)"},
}
MODEL_CACHE_DIR = DATA_DIR / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

_download_state = {"tier": None, "progress": 0, "status": "idle", "error": None}


def _tier_local_path(tier: str) -> Path:
    return MODEL_CACHE_DIR / tier


def _tier_is_downloaded(tier: str) -> bool:
    p = _tier_local_path(tier)
    return p.exists() and (p / "config.json").exists()


@app.get("/api/models/available")
def models_available():
    out = {}
    for tier, meta in MODEL_TIERS.items():
        out[tier] = {**meta, "downloaded": _tier_is_downloaded(tier), "local_path": str(_tier_local_path(tier))}
    return out


@app.get("/api/models/download-status")
def models_download_status():
    return _download_state


def _download_tier_background(tier: str):
    meta = MODEL_TIERS[tier]
    _download_state.update({"tier": tier, "status": "downloading", "progress": 0, "error": None})
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=meta["repo"],
            local_dir=str(_tier_local_path(tier)),
            local_dir_use_symlinks=False,
        )
        _download_state.update({"status": "done", "progress": 100})
    except Exception as exc:
        _download_state.update({"status": "error", "error": str(exc)})


@app.post("/api/models/download/{tier}")
def models_download(tier: str):
    if tier not in MODEL_TIERS:
        return {"error": f"unknown tier; options: {list(MODEL_TIERS.keys())}"}
    if _tier_is_downloaded(tier):
        return {"tier": tier, "already_downloaded": True, "path": str(_tier_local_path(tier))}
    if _download_state["status"] == "downloading":
        return {"error": f"already downloading {_download_state['tier']}"}
    threading.Thread(target=_download_tier_background, args=(tier,), daemon=True).start()
    return {"tier": tier, "started": True, "repo": MODEL_TIERS[tier]["repo"]}


if __name__ == "__main__":
    print(f"Outlier Desktop {APP_VERSION} starting on http://127.0.0.1:8765")
    if SKIP_MODEL:
        print("  (stub mode — set OUTLIER_SKIP_MODEL=0 or unset to load model)")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
