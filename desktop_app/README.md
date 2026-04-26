# Outlier Desktop v0.1 (Stub Mode)

FastAPI + vanilla HTML/JS desktop chat interface for Outlier.

## Run
```bash
bash scripts/launch.sh
```
Opens http://127.0.0.1:8765

## Features (v0.1)
- [x] Chat UI with streaming responses
- [x] Profile switcher (Default/Medical/Coding/Legal)
- [x] History persisted to ~/.outlier/history.jsonl
- [x] Profile alphas persisted to ~/.outlier/profiles.json
- [x] Health check + version API
- [ ] Real model inference (v0.2)
- [ ] Profile training (v0.3)

## Migration to v0.2
Replace `stream_stub_response()` in `backend/server.py` with subprocess call to outlier-engine.
