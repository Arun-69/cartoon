"""FastAPI backend: job API + serves the frontend.
Run:  uvicorn backend.main:app --port 7860   (from the project root)
"""
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
JOB_TTL = 3600            # 1 hour-ku apram videos delete aagum
MAX_MUSIC_MB = 10
MAX_SONG_MB = 25
MAX_IMAGE_MB = 12
MAX_IMAGES = 6

app = FastAPI(title="Kutti Padam Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pool = ThreadPoolExecutor(max_workers=2)   # free CPU-ku 2 podhum
JOBS = {}


def cleanup():
    now = time.time()
    for jid in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL]:
        shutil.rmtree(JOBS[jid]["workdir"], ignore_errors=True)
        del JOBS[jid]


# main.py — replace the run_job function and pool.submit call

def run_job(jid, args):
    job = JOBS[jid]
    job["status"] = "running"

    def on_progress(p, msg):
        job["progress"], job["message"] = p, msg

    try:
        if args.get("song_file"):
            # Song/hero pipeline lives in the other module
            from . import song_pipeline          # <-- rename your second file to this
            res = song_pipeline.run_song(
                topic=args["topic"],
                song_path=args["song_file"],
                lyrics=args["lyrics"],
                use_subs=args["use_subs"],
                workdir=job["workdir"],
                on_progress=on_progress,
            )
        else:
            # Story/rhyme pipeline — note: no song_file / lyrics / image_files
            res = pipeline.run_pipeline(
                topic=args["topic"],
                voice_label=args["voice_label"],
                n_scenes=args["n_scenes"],
                mode=args["mode"],
                use_subs=args["use_subs"],
                music_choice=args["music_choice"],
                music_file=args["music_file"],
                workdir=job["workdir"],
                on_progress=on_progress,
            )
        job.update(status="done", progress=1.0, message="Ready!", title=res["title"],
                   scenes=res["scenes"], video=res["video"])
    except Exception as e:
        job.update(status="error", error=str(e))


@app.get("/api/health")
def health():
    return {"ok": True}


def infer_settings(topic):
    """Prompt-la irundhu type + voice-a guess pannum (user prompt mattum kudutha podhum)."""
    t = topic.lower()
    rhyme_words = ("rhyme", "song", "lullaby", "paatu", "pattu", "paadal", "thalattu",
                   "பாட்டு", "பாடல்", "தாலாட்டு")
    mode = pipeline.MODES[1] if any(w in t for w in rhyme_words) else pipeline.MODES[0]
    if "hindi" in t or "हिंदी" in topic:
        voice = "Hindi - Female"
    elif "english" in t:
        voice = "English (India) - Female"
    else:
        voice = "Tamil - Female (Pallavi)"
    return voice, mode


@app.get("/api/options")
def options():
    return {"voices": list(pipeline.VOICES), "modes": pipeline.MODES,
            "music": pipeline.MUSIC_CHOICES}


@app.post("/api/jobs")
async def create_job(topic: str = Form(""), voice: Optional[str] = Form(None),
                     mode: Optional[str] = Form(None), music: Optional[str] = Form(None),
                     scenes: int = Form(5), subtitles: bool = Form(True),
                     music_file: Optional[UploadFile] = File(None),
                     song_file: Optional[UploadFile] = File(None), lyrics: str = Form(""),
                     images: List[UploadFile] = File(default=[])):
    cleanup()
    topic = topic.strip()
    has_media = bool((song_file and song_file.filename) or any(f.filename for f in images))
    if len(topic) > 10000 or not (topic or has_media):
        raise HTTPException(400, "Topic type pannunga (10000 characters varai), illa image/paatu upload pannunga.")
    guess_voice, guess_mode = infer_settings(topic)
    voice, mode = voice or guess_voice, mode or guess_mode
    music = music or pipeline.MUSIC_CHOICES[1]
    if voice not in pipeline.VOICES or mode not in pipeline.MODES or music not in pipeline.MUSIC_CHOICES:
        raise HTTPException(400, "Thappaana option.")
    try:
        pipeline.check_quota()
    except pipeline.QuotaError as e:
        raise HTTPException(429, str(e))

    workdir = tempfile.mkdtemp()
    music_path = None
    if music_file and music_file.filename:
        ext = Path(music_file.filename).suffix.lower()
        data = await music_file.read()
        if ext not in {".mp3", ".wav", ".m4a", ".ogg"} or len(data) > MAX_MUSIC_MB * 1024 * 1024:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, f"Music mp3/wav/m4a/ogg, {MAX_MUSIC_MB}MB-kulla irukkanum.")
        music_path = os.path.join(workdir, "own_music" + ext)
        with open(music_path, "wb") as f:
            f.write(data)

    song_path = None
    if song_file and song_file.filename:
        ext = Path(song_file.filename).suffix.lower()
        data = await song_file.read()
        if ext not in {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac"} or len(data) > MAX_SONG_MB * 1024 * 1024:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, f"Paatu mp3/wav/m4a/ogg/aac/flac, {MAX_SONG_MB}MB-kulla irukkanum.")
        song_path = os.path.join(workdir, "user_song" + ext)
        with open(song_path, "wb") as f:
            f.write(data)

    image_paths = []
    for k, up in enumerate([f for f in images if f.filename][:MAX_IMAGES]):
        ext = Path(up.filename).suffix.lower()
        data = await up.read()
        if ext not in {".png", ".jpg", ".jpeg", ".webp"} or len(data) > MAX_IMAGE_MB * 1024 * 1024:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, f"Image png/jpg/webp, {MAX_IMAGE_MB}MB-kulla irukkanum.")
        ip = os.path.join(workdir, f"upload_{k}{ext}")
        with open(ip, "wb") as f:
            f.write(data)
        image_paths.append(ip)

    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"status": "queued", "progress": 0.0, "message": "Line-la irukku...",
                 "created": time.time(), "workdir": workdir}
    pool.submit(run_job, jid, dict(
        topic=topic,
        voice_label=voice,
        n_scenes=max(3, min(8, scenes)),
        mode=mode,
        use_subs=subtitles,
        music_choice=music,
        music_file=music_path,
        song_file=song_path,
        lyrics=lyrics.strip()[:4000],
        image_files=image_paths,
    ))
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Job kedaikala (expire aagiruchu irukkalaam).")
    out = {k: job.get(k) for k in ("status", "progress", "message", "title", "scenes", "error")}
    if job["status"] == "done":
        out["video_url"] = f"/api/jobs/{jid}/video"
    return out


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str, download: bool = False):
    job = JOBS.get(jid)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Video ready illa.")
    return FileResponse(job["video"], media_type="video/mp4", filename="kutti-padam.mp4",
                        content_disposition_type="attachment" if download else "inline")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="web")