---
title: Kutti Padam Studio
sdk: docker
app_port: 7860
---

# Kutti Padam Studio
FastAPI backend (`backend/`) + plain HTML/CSS/JS frontend (`frontend/`).

Run locally (ffmpeg venum):
    pip install -r requirements.txt
    uvicorn backend.main:app --port 7860
Open http://localhost:7860

Env vars: GEMINI_API_KEY, MAX_VIDEOS_PER_DAY (default 30), SUB_FONT.
