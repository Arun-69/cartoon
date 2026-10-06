"""
Kids video pipeline (no UI)
Prompt -> Story/Rhyme (Gemini) -> 3D-style images (Pollinations) -> Voice (edge-tts)
       -> Animation -> Subtitles + Music -> MP4

Animation priority (ellaame free, paid illa):
  1. HF_SPACE set pannina  -> free AI image-to-video (Hugging Face Space)      [optional]
  2. anim3d.py             -> free pseudo-3D engine (depth parallax + hop + sparkles) [default]
  3. old puppet wobble     -> fallback
  4. Ken-Burns camera move -> last fallback
FAL_KEY (paid) code irukku, aana FAL_KEY set pannalana off-la dhaan irukkum.
Server run: uvicorn backend.main:app --port 7860   (project root-la irundhu)
"""
import asyncio
import base64
import datetime
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import wave

import edge_tts
import requests
from PIL import Image, ImageDraw

from . import hf_video, imagemode, song

# ---------------- Settings ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")          # free key: aistudio.google.com
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # 404 vandha auto-a vera model kandupidikkum
MAX_VIDEOS_PER_DAY = int(os.getenv("MAX_VIDEOS_PER_DAY", "30"))  # protects free limits
FPS = 25
W, H = 1280, 720
# FREE character animation: rembg-la character-a cut panni, voice-oda sync-a asaiya vaikkum (CPU-laye run aagum)
import importlib.util as _ilu
CHAR_ANIM = (os.getenv("CHAR_ANIM", "auto").lower() != "off"
             and _ilu.find_spec("rembg") is not None and _ilu.find_spec("numpy") is not None)
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")   # light/fast venum-na "u2netp"
# Real 3D animation (image-to-video) - optional, paid per second. FAL_KEY set pannina automatic-a on.
FAL_KEY = os.getenv("FAL_KEY", "")
FAL_MODEL = os.getenv("FAL_MODEL", "fal-ai/ltx-2.3/image-to-video")
FAL_EXTRA = json.loads(os.getenv("FAL_EXTRA_JSON", "{}"))   # eg {"duration": 5}
ANIMATE = bool(FAL_KEY) and os.getenv("ANIMATE", "auto").lower() != "off"
# FREE pseudo-3D engine (anim3d.py): rembg + opencv + scipy irundha automatic-a on
ANIM3D = (os.getenv("ANIM3D", "auto").lower() != "off"
          and all(_ilu.find_spec(m) is not None for m in ("rembg", "numpy", "cv2", "scipy")))
# Channel-style: ONE hero character (consistent) + scene-ku thaniya background. anim3d irundha automatic-a on.
HERO_MODE = os.getenv("HERO_MODE", "auto").lower() != "off"
SUB_FONT = os.getenv("SUB_FONT", "Noto Sans Tamil")
SFX = os.getenv("SFX", "on").lower() != "off"         # boing / ting sound effects (3D clips)
CARDS = os.getenv("CARDS", "on").lower() != "off"     # rainbow intro + "Bye Bye" outro
VOICE_PITCH = os.getenv("VOICE_PITCH", "").strip()    # eg "+10Hz"; "+0Hz" = off  # Hindi-ku "Noto Sans Devanagari"

# label -> (language, edge-tts voice)
VOICES = {
    "Tamil - Female (Pallavi)": ("Tamil", "ta-IN-PallaviNeural"),
    "Tamil - Male (Valluvar)": ("Tamil", "ta-IN-ValluvarNeural"),
    "English (India) - Female": ("English", "en-IN-NeerjaNeural"),
    "English (India) - Male": ("English", "en-IN-PrabhatNeural"),
    "Hindi - Female": ("Hindi", "hi-IN-SwaraNeural"),
}

MODES = ["Story (kadhai)", "Rhyme (paatu)"]
MUSIC_CHOICES = ["No music", "Cartoon fun tune (bouncy)", "Soft music-box tune"]

STYLE = ("cute colorful kids cartoon, soft 3D cartoon render like popular nursery-rhyme videos, "
         "ONE round chubby adorable main character centered, full body visible from head to toe "
         "standing on the ground, facing the camera, huge shiny eyes, big happy smile, "
         "bright rainbow candy colors, simple cheerful background with clouds flowers and hills, "
         "soft lighting, smooth shading, child-friendly, no text, no watermark, ")

_usage = {"day": datetime.date.today(), "count": 0}



# ---------------- Gemini helper (auto model fallback) ----------------
_GEM = "https://generativelanguage.googleapis.com/v1beta"
_good_model = {"name": None}


def _gem_headers():
    return {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}


def _flash_candidates():
    """Ungal key-ku kedaikkura flash models-a list pannum, puthusa irukkuradhu mudhal."""
    r = requests.get(f"{_GEM}/models?pageSize=200", headers=_gem_headers(), timeout=30)
    r.raise_for_status()
    skip = ("tts", "image", "live", "audio", "embed", "omni", "robotics", "computer", "thinking")
    names = []
    for m in r.json().get("models", []):
        n = m["name"].replace("models/", "")
        if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n \
                and not any(x in n for x in skip):
            names.append(n)

    def key(n):
        v = re.search(r"(\d+(?:\.\d+)?)", n)
        return (float(v.group(1)) if v else 0.0, "lite" not in n)
    return sorted(set(names), key=key, reverse=True)


def gemini_generate(prompt, audio_path=None, image_paths=None):
    parts_in = [{"text": prompt}]
    for ip in image_paths or []:   # images-a Gemini paathu narration ezhudha
        with open(ip, "rb") as f:
            parts_in.insert(len(parts_in) - 1, {"inline_data": {"mime_type": "image/jpeg",
                                                               "data": base64.b64encode(f.read()).decode()}})
    if audio_path:   # paatu audio-va Gemini-kku anuppum (lyrics + timing kandupidikka)
        with open(audio_path, "rb") as f:
            parts_in.insert(0, {"inline_data": {"mime_type": "audio/mpeg",
                                                "data": base64.b64encode(f.read()).decode()}})
    body = {"contents": [{"parts": parts_in}],
            "generationConfig": {"responseMimeType": "application/json"}}
    models = [_good_model["name"] or GEMINI_MODEL]
    tried = []
    for attempt in range(2):
        for m in models:
            r = requests.post(f"{_GEM}/models/{m}:generateContent",
                              headers=_gem_headers(), json=body, timeout=180 if audio_path else 90)
            if r.status_code == 200:
                _good_model["name"] = m
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts if not p.get("thought"))
            tried.append(f"{m}:{r.status_code}")
            if r.status_code in (400, 401, 403) and "key" in r.text.lower():
                raise RuntimeError("Gemini API key thappu / permission illa. Puthu key create pannunga.")
        if attempt == 0:  # configured model work aagala -> available models-a theduvom
            try:
                models = [m for m in _flash_candidates()[:6] if m not in models]
            except Exception:
                break
    raise RuntimeError("Gemini model work aagala (" + ", ".join(tried) + "). GEMINI_MODEL env-la vera model name kudunga.")

# ---------------- Step 1: Story / Rhyme ----------------
def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Story JSON kedaikala")
    return json.loads(text[start:end + 1])


def make_story(topic, language, n_scenes, mode):
    if mode.startswith("Rhyme"):
        kind = f"""Write a short, cheerful RHYMING kids song (nursery rhyme style), age 2-6.
Each scene "narration" = 2 short rhyming lines in {language}, simple words, with a repeating,
catchy feel (the last scene can repeat the first line)."""
    else:
        kind = f"""Write a short, positive kids story, age 3-8.
Each scene "narration" = 1-2 simple sentences in {language}."""

    prompt = f"""{kind}
Topic: {topic}
Number of scenes: {n_scenes}

Rules:
- Each scene "image_prompt": ENGLISH only, describe the scene visually.
  Repeat the same character descriptions (color, clothes, look) in EVERY image_prompt
  so characters look consistent.
- "character": ENGLISH, ONE main cartoon character (animal or child) that appears in every scene;
  one sentence (species, colors, clothes).
- Each scene "background_prompt": ENGLISH, the SAME place as image_prompt but EMPTY:
  no characters, no people, no animals.
- Each scene "motion_prompt": ENGLISH, one short sentence about what moves
  (e.g. "the crow flaps its wings and hops, camera slowly pushes in").
- ONE cute main character (animal or child) in every scene, full body, cartoon look,
  happy faces, bright colors. Keep backgrounds simple and cheerful.
- No violence or scary content.

Return ONLY JSON in this format:
{{"title": "...", "character": "...", "scenes": [{{"narration": "...", "image_prompt": "...", "background_prompt": "...", "motion_prompt": "..."}}]}}"""

    text = None
    if GEMINI_API_KEY:
        try:
            text = gemini_generate(prompt)
        except RuntimeError as e:
            if "key" in str(e).lower() and "thappu" in str(e):
                raise
            gemini_err = str(e)
    if text is None:
        # Key illa / Gemini fail -> keyless fallback (less reliable)
        try:
            r = requests.get("https://text.pollinations.ai/" + urllib.parse.quote(prompt), timeout=90)
            r.raise_for_status()
            text = r.text
        except Exception:
            raise RuntimeError(locals().get("gemini_err") or "Story service work aagala. GEMINI_API_KEY check pannunga.")

    story = extract_json(text)
    scenes = story.get("scenes", [])[:n_scenes]
    if not scenes:
        raise ValueError("Scenes varala, marubadiyum try pannunga")
    return story.get("title", "Kids Video"), scenes, story.get("character")


# ---------------- Step 2: Images ----------------
def placeholder(path, text):
    img = Image.new("RGB", (1024, 576), (255, 214, 153))
    d = ImageDraw.Draw(img)
    d.text((40, 260), text[:90], fill=(80, 40, 0))
    img.save(path, "JPEG")


IMG_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()   # free: enter.pollinations.ai
IMG_GAP = float(os.getenv("IMG_MIN_GAP", "0" if IMG_KEY else "16"))  # anonymous = 15s-ku oru request
_img_lock = threading.Lock()
_img_last = [0.0]


def make_image(image_prompt, path, seed, style=None):
    """True = real image kedaichuduchu, False = placeholder (orange screen) potaachu."""
    full = urllib.parse.quote((STYLE if style is None else style) + image_prompt)
    if IMG_KEY:
        url = (f"https://gen.pollinations.ai/image/{full}"
               f"?width=1024&height=576&seed={seed}&nologo=true&key={IMG_KEY}")
    else:
        url = (f"https://image.pollinations.ai/prompt/{full}"
               f"?width=1024&height=576&model=flux&nologo=true&seed={seed}")
    for attempt in range(4):
        with _img_lock:   # request-kalukku naduvula gap (anonymous throttle)
            wait = IMG_GAP - (time.time() - _img_last[0])
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(url, timeout=120)
                _img_last[0] = time.time()
                if r.status_code == 200 and r.content:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    Image.open(path).convert("RGB").save(path, "JPEG")  # validate
                    return True
                print(f"[image] attempt {attempt + 1}/4 HTTP {r.status_code}: {r.text[:120]!r}")
            except Exception as e:
                _img_last[0] = time.time()
                print(f"[image] attempt {attempt + 1}/4 error: {e!r}")
        time.sleep(5 + attempt * 5)
    print("[image] FAILED -> orange placeholder. POLLINATIONS_API_KEY set pannunga (free).")
    placeholder(path, image_prompt)
    return False


# ---------------- Step 3: Voice ----------------
def speak(text, voice, path, rate, pitch=None, silent_dur=4.0):
    """Text irundha voice; illa-na (image mode, Gemini key illa) silent audio."""
    if not (text or "").strip():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", str(silent_dur), "-c:a", "libmp3lame", path], check=True, capture_output=True)
        return
    make_voice(text, voice, path, rate, pitch)


def make_voice(text, voice, path, rate, pitch=None):
    """pitch eg "+10Hz" -> konjam kutti/cheerful voice. Pitch work aagala-na pitch illama retry."""
    async def _run(p):
        kw = {"rate": rate}
        if p:
            kw["pitch"] = p
        await edge_tts.Communicate(text, voice, **kw).save(path)

    if pitch and pitch != "+0Hz":
        try:
            asyncio.run(_run(pitch))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return
        except Exception as e:
            print(f"[voice] pitch {pitch} fail ({e!r}) -> pitch illama retry")
    asyncio.run(_run(None))


# ---------------- Step 4: Music (generated, copyright-free) ----------------
def make_music_box(path):
    """Twinkle-style music box tune, synthesized (no copyright issue)."""
    sr = 44100
    notes = {"C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23, "G": 392.00, "A": 440.00}
    melody = ("CCGGAAG" "FFEEDDC" "GGFFEED" "GGFFEED" "CCGGAAG" "FFEEDDC")
    beat = 0.55
    samples = []
    for ch in melody:
        f = notes[ch]
        n = int(sr * beat)
        for i in range(n):
            t = i / sr
            env = math.exp(-3.5 * t)
            v = (math.sin(2 * math.pi * f * t)
                 + 0.35 * math.sin(2 * math.pi * 2 * f * t)
                 + 0.1 * math.sin(2 * math.pi * 3 * f * t))
            samples.append(0.45 * env * v)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32767)) for s in samples))


# ---------------- Step 5: Video ----------------
def audio_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path])
    return float(out.decode().strip())


def srt_time(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
    return f"{h:02d}:{m:02d}:{int(sec):02d},{int((sec - int(sec)) * 1000):03d}"


def animate_image(img, motion, out):
    """Image -> real animated clip (fal.ai queue API). Paid per second of video."""
    with open(img, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    body = {"image_url": "data:image/jpeg;base64," + data, "prompt": motion}
    body.update(FAL_EXTRA)
    h = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    r = requests.post(f"https://queue.fal.run/{FAL_MODEL}", headers=h, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"fal submit {r.status_code}: {r.text[:300]}")
    q = r.json()
    t0 = time.time()
    while True:
        st = requests.get(q["status_url"], headers=h, timeout=30).json()
        if st.get("status") == "COMPLETED":
            break
        if st.get("status") in ("FAILED", "ERROR") or time.time() - t0 > 420:
            raise RuntimeError(f"fal status: {st}")
        time.sleep(3)
    res = requests.get(q["response_url"], headers=h, timeout=30).json()
    v = requests.get(res["video"]["url"], timeout=180)
    v.raise_for_status()
    with open(out, "wb") as f:
        f.write(v.content)


def make_pingpong(video, out, workdir):
    """Clip short-a irundha, forward+reverse pannina smooth-a loop aagum."""
    fc = (f"[0:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps={FPS},"
          "split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[v]")
    subprocess.run(["ffmpeg", "-y", "-i", video, "-filter_complex", fc, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out],
                   check=True, capture_output=True, cwd=workdir)


def camera_move(idx, frames):
    """Image-ku 4 vidha camera movement (zoom in / zoom out / pan right / pan left)."""
    c = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    k = idx % 4
    if k == 0:
        z = "z='min(zoom+0.0009,1.25)'"
    elif k == 1:
        z = "z='if(eq(on,0),1.25,max(zoom-0.0009,1.0))'"
    elif k == 2:
        return f"z='1.2':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'"
    else:
        return f"z='1.2':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-(ih/zoom/2)'"
    return f"{z}:{c}"


def make_clip(img, audio, out, workdir, idx, sub_text=None, video=None, pad=0.5):
    dur = audio_duration(audio) + pad
    frames = int(dur * FPS)
    if video:
        pp = os.path.join(workdir, f"pp{idx}.mp4")
        make_pingpong(video, pp, workdir)
        inputs = ["-stream_loop", "-1", "-i", pp, "-i", audio]
        base = "setsar=1"
    else:
        inputs = ["-i", img, "-i", audio]
        base = (f"scale=1920:1080,zoompan={camera_move(idx, frames)}:d={frames}"
                f":s=1280x720:fps={FPS}")

    def run(vf):
        subprocess.run([
            "ffmpeg", "-y", *inputs, "-vf", vf, "-af", "apad", "-map", "0:v", "-map", "1:a",
            "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", out],
            check=True, capture_output=True, cwd=workdir)

    if sub_text:
        srt = f"sub{idx}.srt"
        with open(os.path.join(workdir, srt), "w", encoding="utf-8") as f:
            f.write(f"1\n00:00:00,000 --> {srt_time(dur - 0.1)}\n{sub_text.strip()}\n")
        style = f"FontName={SUB_FONT},FontSize=16,Outline=2,Shadow=0,Alignment=2,MarginV=25"
        try:
            run(f"{base},subtitles={srt}:force_style='{style}',format=yuv420p")
            return
        except subprocess.CalledProcessError:
            pass  # subtitles fail aanaa, subtitle illama continue
    run(base + ",format=yuv420p")


# ---------------- FREE character animation (cutout + voice-sync motion) ----------------
_rembg_session = {}


def fill_background(im, alpha):
    """Character irundha idatha, suthi irukkura background-a vechu nirappum (asaiyumbodhu ghost varaadhu)."""
    import numpy as np
    from PIL import ImageFilter, ImageOps
    mask = alpha.point(lambda p: 255 if p > 20 else 0).filter(ImageFilter.MaxFilter(21))
    known = ImageOps.invert(mask)
    black = Image.new("RGB", im.size, (0, 0, 0))
    base = Image.composite(im, black, known)
    out = np.asarray(im).astype(np.float32)
    unknown = np.asarray(mask) > 127
    filled = ~unknown
    for sigma, thr in ((12, 0.05), (36, 0.05), (90, 0.03), (250, 0.01)):
        num = np.asarray(base.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32)
        den = np.asarray(known.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32) / 255.0
        ok = unknown & ~filled & (den > thr)
        out[ok] = num[ok] / den[ok][:, None]
        filled |= ok
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")


_seg_cache = {}


def _segment(img_path):
    """rembg run (oru image-ku oru thadavai mattum). Returns (im, fg_rgba, alpha) illa None."""
    import numpy as np
    from rembg import new_session, remove
    if img_path in _seg_cache:
        return _seg_cache[img_path]
    if REMBG_MODEL not in _rembg_session:
        _rembg_session[REMBG_MODEL] = new_session(REMBG_MODEL)
    im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
    fg = remove(im, session=_rembg_session[REMBG_MODEL]).convert("RGBA")
    alpha = fg.getchannel("A")
    frac = float((np.asarray(alpha) > 40).mean())
    res = None if (frac < 0.03 or frac > 0.85) else (im, fg, alpha)
    _seg_cache.clear()
    _seg_cache[img_path] = res
    return res


def extract_layers3d(img_path):
    """Free 3D engine-ku layers (background + character + depth). None = character kedaikala."""
    from . import anim3d
    seg = _segment(img_path)
    if not seg:
        return None
    im, fg, alpha = seg
    return anim3d.build_layers(im, alpha, fill_background(im, alpha), W, H)


def extract_cutout(img_path):
    """Returns (background, character_rgba, anchor) illa None (character kedaikala-na)."""
    seg = _segment(img_path)
    if not seg:
        return None
    im, fg, alpha = seg
    bbox = alpha.point(lambda p: 255 if p > 40 else 0).getbbox()
    anchor = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))     # paadham (feet) pivot
    return fill_background(im, alpha), fg, anchor


def audio_envelope(audio, frames):
    """Voice loudness per video frame (0..1) - character pesumbodhu asaiya."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    spf = 8000 // FPS
    n = len(x) // spf
    out = np.zeros(frames, dtype=np.float32)
    if n:
        rms = np.sqrt((x[:n * spf].reshape(n, spf) ** 2).mean(axis=1))
        env = np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1)
        env = np.convolve(env, np.ones(3) / 3, mode="same")
        out[:min(frames, n)] = env[:frames]
    return out


def _affine(fg, anchor, s, theta, dx, dy):
    ax, ay = anchor
    c, sn = math.cos(theta), math.sin(theta)
    a, b = c / s, sn / s
    cc = ax + (-c * (ax + dx) - sn * (ay + dy)) / s
    d, e = -sn / s, c / s
    f = ay + (sn * (ax + dx) - c * (ay + dy)) / s
    return fg.transform((W, H), Image.AFFINE, (a, b, cc, d, e, f), resample=Image.BILINEAR)


def render_frames(cut, env, frames, idx):
    bg, fg, anchor = cut
    zoom_in = idx % 2 == 0
    for n in range(frames):
        t, e = n / FPS, float(env[n])
        breathe = 1 + 0.012 * math.sin(2 * math.pi * 0.45 * t)
        s = breathe * (1 + 0.035 * e)                                   # pesumbodhu konjam periya aagum
        theta = math.radians(1.6) * math.sin(2 * math.pi * 0.55 * t + idx) \
            + math.radians(2.0) * e * math.sin(2 * math.pi * 3.0 * t)    # sway + pesum wiggle
        dx = 6 * math.sin(2 * math.pi * 0.35 * t + idx)
        dy = -12 * e                                                     # voice-ku thagundha bounce
        frame = bg.copy()
        fgt = _affine(fg, anchor, s, theta, dx, dy)
        frame.paste(fgt, (0, 0), fgt)
        k = n / max(1, frames - 1)
        z = 1 + 0.07 * (k if zoom_in else 1 - k)                         # slow camera zoom
        cw, ch = W / z, H / z
        x0, y0 = (W - cw) / 2, (H - ch) / 2
        yield frame.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)).tobytes()


def _encode_frames(make_gen, audio, dur, out, workdir, idx, sub_text, sub_vf=None):
    """Raw RGB frames (make_gen() -> iterator of bytes) + audio + subtitles -> mp4."""
    def encode(vf):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio, "-vf", vf,
               "-af", "apad", "-map", "0:v", "-map", "1:a", "-t", f"{dur:.2f}",
               "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", "44100",
               "-ac", "2", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir)
        try:
            for fr in make_gen():
                p.stdin.write(fr)
        except BrokenPipeError:
            pass
        finally:
            try:
                p.stdin.close()
            except Exception:
                pass
        err = p.stderr.read()
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd, stderr=err)

    if sub_text:
        srt = f"sub{idx}.srt"
        with open(os.path.join(workdir, srt), "w", encoding="utf-8") as f:
            f.write(f"1\n00:00:00,000 --> {srt_time(dur - 0.1)}\n{sub_text.strip()}\n")
        style = f"FontName={SUB_FONT},FontSize=16,Outline=2,Shadow=0,Alignment=2,MarginV=25"
        sub_vf = f"subtitles={srt}:force_style='{style}'"
    if sub_vf:
        try:
            encode(f"{sub_vf},format=yuv420p")
            return
        except subprocess.CalledProcessError:
            pass  # text render aagala-na text illama continue
    encode("format=yuv420p")


def make_character_clip(cut, audio, out, workdir, idx, sub_text=None):
    """OLD puppet-wobble fallback."""
    dur = audio_duration(audio) + 0.5
    frames = int(dur * FPS)
    env = audio_envelope(audio, frames)
    _encode_frames(lambda: render_frames(cut, env, frames, idx), audio, dur, out, workdir, idx, sub_text)


def make_3d_clip(layers, audio, out, workdir, idx, sub_text=None, rhyme=False):
    """FREE pseudo-3D clip (anim3d.py)."""
    from . import anim3d, cartoon_audio
    dur = audio_duration(audio) + 0.5
    frames = int(dur * FPS)
    env = audio_envelope(audio, frames)
    mixed = audio
    if SFX:  # boing (landing) + ting (sparkle), hop-oda exact sync
        try:
            sfx = os.path.join(workdir, f"sfx{idx}.wav")
            cartoon_audio.make_scene_sfx(sfx, dur, rhyme, idx, env, FPS)
            mixed = os.path.join(workdir, f"mix{idx}.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio, "-i", sfx, "-filter_complex",
                            "[0:a]apad[a0];[a0][1:a]amix=inputs=2:duration=shortest:normalize=0[a]",
                            "-map", "[a]", "-ar", "44100", "-ac", "2", mixed],
                           check=True, capture_output=True)
        except Exception as e:
            print(f"[sfx] scene {idx + 1} fail: {e!r}")
            mixed = audio
    _encode_frames(lambda: anim3d.render_frames(layers, env, frames, idx, FPS, rhyme),
                   mixed, dur, out, workdir, idx, sub_text)


def _ass_time(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def make_card(text, kind, out, workdir, dur):
    """Rainbow title card (intro) / 'Bye Bye' card (outro) + jingle."""
    from . import anim3d, cartoon_audio
    jingle = os.path.join(workdir, f"jingle_{kind}.wav")
    cartoon_audio.make_jingle(jingle, dur, kind)
    clean = re.sub(r"[{}\\]", "", text).replace("\n", "\\N")
    align, size, mv = (8, 60, 60) if kind == "intro" else (5, 96, 0)
    ass = f"card_{kind}.ass"
    with open(os.path.join(workdir, ass), "w", encoding="utf-8") as f:
        f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Card,{SUB_FONT},{size},&H0050E6FF,&H0050E6FF,&H00A02878,&H64000000,-1,0,0,0,100,100,0,0,1,6,3,{align},60,60,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{_ass_time(0.1)},{_ass_time(dur)},Card,,0,0,0,,{{\\fad(400,300)\\t(0,500,\\fscx112\\fscy112)\\t(500,1000,\\fscx100\\fscy100)}}{clean}
""")
    frames = int(dur * FPS)
    _encode_frames(lambda: anim3d.render_card_frames(W, H, frames, FPS, 3 if kind == "intro" else 5),
                   jingle, dur, out, workdir, 900 + (kind == "outro"), None, sub_vf=f"ass={ass}")


def join_clips(clips, out, workdir):
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", out], check=True, capture_output=True)


def add_music(video, music, out, duck=True):
    """Voice pesumbodhu music thaanaa kuraiyum (ducking). Ducking fail aana simple mix."""
    base = ["ffmpeg", "-y", "-i", video, "-stream_loop", "-1", "-i", music, "-filter_complex"]
    tail = ["-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", out]
    if duck:
        try:
            subprocess.run(base + [
                "[0:a]asplit=2[v1][v2];[1:a]volume=0.55[m];"
                "[m][v1]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=400[md];"
                "[v2][md]amix=inputs=2:duration=first:normalize=0[a]"] + tail,
                check=True, capture_output=True)
            return
        except subprocess.CalledProcessError:
            print("[music] ducking fail -> simple mix")
    subprocess.run(base + [
        "[1:a]volume=0.18[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[a]"] + tail,
        check=True, capture_output=True)


# ---------------- Runner ----------------
class QuotaError(Exception):
    pass


def check_quota():
    today = datetime.date.today()
    if _usage["day"] != today:
        _usage.update(day=today, count=0)
    if _usage["count"] >= MAX_VIDEOS_PER_DAY:
        raise QuotaError("Inniku free limit mudinjiduchu. Naalaikku try pannunga.")


def run_pipeline(topic, voice_label, n_scenes, mode, use_subs, music_choice,
                 music_file, workdir, on_progress, song_file=None, lyrics="", image_files=None):
    """Returns {"title", "scenes", "video"}. on_progress(fraction, message)."""
    if song_file:   # Song mode: user kudutha paatu thaan main audio
        return song.run_song(topic, song_file, lyrics, use_subs, workdir, on_progress)
    language, voice = VOICES[voice_label]
    rate = "-12%" if mode.startswith("Rhyme") else "-5%"
    pitch = VOICE_PITCH or ("+10Hz" if mode.startswith("Rhyme") else "+4Hz")
    try:
        on_progress(0.05, "Kadhai ezhudhuren...")
        image_files = list(image_files or [])
        if image_files:   # IMAGE mode: ungal image(s) appadiye animate aagum
            on_progress(0.04, "Images-a paakkuren (storyboard-na pirikkuren)...")
            image_files = imagemode.prepare_images(image_files, workdir)
            title, scenes = imagemode.narrate(image_files, topic, language, mode)
            character = None
        else:
            title, scenes, character = make_story(topic, language, n_scenes, mode)

        hero = None   # ONE consistent main character, scene-ku scene background maarum
        if HERO_MODE and ANIM3D and character:
            on_progress(0.07, "Main character create panren...")
            try:
                hero = song.make_hero(character, workdir)
            except Exception as e:
                print(f"[hero] fail: {e}")

        clips = []
        for i, sc in enumerate(scenes):
            on_progress(0.1 + 0.75 * i / len(scenes), f"Scene {i + 1}/{len(scenes)} ready aaguthu...")
            img = os.path.join(workdir, f"img{i}.jpg")
            aud = os.path.join(workdir, f"aud{i}.mp3")
            clip = os.path.join(workdir, f"clip{i}.mp4")
            if hero is not None and sc.get("background_prompt"):
                try:
                    make_image(sc["background_prompt"], img, seed=1000 + i, style=song.BG_STYLE)
                    speak(sc["narration"], voice, aud, rate, pitch)
                    layers = song.hero_layers(img, hero, i)
                    make_3d_clip(layers, aud, clip, workdir, i, sc["narration"] if use_subs else None,
                                 rhyme=mode.startswith("Rhyme"))
                    clips.append(clip)
                    continue
                except Exception as e:  # fail aana per-scene image flow-ku povum
                    print(f"[hero] scene {i + 1} fail: {e}")
            if image_files:
                shutil.copyfile(image_files[i], img)
            elif not make_image(sc["image_prompt"], img, seed=1000 + i):
                on_progress(0.1 + 0.75 * i / len(scenes),
                            f"WARNING Scene {i + 1}: image generate aagala (Pollinations) - placeholder use aaguthu")
            speak(sc["narration"], voice, aud, rate, pitch)
            vid = None
            if hf_video.available():
                on_progress(0.1 + 0.75 * (i + 0.5) / len(scenes),
                            f"Scene {i + 1}/{len(scenes)}: free AI animation (HF Space) - konjam neram aagum...")
                try:
                    vid = os.path.join(workdir, f"vid{i}.mp4")
                    hf_video.animate(img, sc.get("motion_prompt") or sc["image_prompt"], vid)
                except Exception as e:  # quota/error -> free 3D engine-ku povum
                    print(f"[hf-video] scene {i + 1} fail: {e}")
                    vid = None
            if vid is None and ANIMATE:
                on_progress(0.1 + 0.75 * (i + 0.5) / len(scenes),
                            f"Scene {i + 1}/{len(scenes)} 3D animate aaguthu (konjam neram aagum)...")
                try:
                    vid = os.path.join(workdir, f"vid{i}.mp4")
                    animate_image(img, sc.get("motion_prompt") or sc["image_prompt"], vid)
                except Exception as e:  # animation fail -> image + camera move
                    print(f"[animate] scene {i + 1} fail: {e}")
                    vid = None
            sub = sc["narration"] if use_subs else None
            done = False
            if vid is None and ANIM3D:
                on_progress(0.1 + 0.75 * (i + 0.6) / len(scenes),
                            f"Scene {i + 1}/{len(scenes)}: 3D animation render aaguthu...")
                try:
                    layers = extract_layers3d(img)
                    if layers:
                        make_3d_clip(layers, aud, clip, workdir, i, sub, rhyme=mode.startswith("Rhyme"))
                        done = True
                except Exception as e:  # fail aana pazhaya puppet mode-ku povum
                    print(f"[anim3d] scene {i + 1} fail: {e}")
            if vid is None and not done and CHAR_ANIM:
                on_progress(0.1 + 0.75 * (i + 0.6) / len(scenes),
                            f"Scene {i + 1}/{len(scenes)}: characters-a asaiya vaikkuren...")
                try:
                    cut = extract_cutout(img)
                    if cut:
                        make_character_clip(cut, aud, clip, workdir, i, sub)
                        done = True
                except Exception as e:  # fail aana camera-move mode-ku povum
                    print(f"[char-anim] scene {i + 1} fail: {e}")
            if not done:
                make_clip(img, aud, clip, workdir, i, sub, vid)
            clips.append(clip)

        if CARDS:
            on_progress(0.88, "Intro / Bye-bye cards...")
            try:
                intro = os.path.join(workdir, "intro.mp4")
                outro = os.path.join(workdir, "outro.mp4")
                make_card(title, "intro", intro, workdir, 3.2)
                make_card("Bye Bye!", "outro", outro, workdir, 2.8)
                clips = [intro] + clips + [outro]
            except Exception as e:  # cards fail aanaalum video tharuvom
                print(f"[cards] fail: {e!r}")
        on_progress(0.9, "Scenes-a join panren...")
        final = os.path.join(workdir, "joined.mp4")
        join_clips(clips, final, workdir)

        music = music_file or None
        if not music and music_choice == MUSIC_CHOICES[1]:
            from . import cartoon_audio
            music = os.path.join(workdir, "tune.wav")
            cartoon_audio.make_cartoon_tune(music)
        elif not music and music_choice == MUSIC_CHOICES[2]:
            music = os.path.join(workdir, "tune.wav")
            make_music_box(music)
        if music:
            on_progress(0.95, "Music serkuren...")
            mixed = os.path.join(workdir, "final.mp4")
            try:
                add_music(final, music, mixed)
                final = mixed
            except subprocess.CalledProcessError:
                pass  # music fail aanaalum video tharuvom
        _usage["count"] += 1
        return {"title": title, "scenes": [s["narration"] for s in scenes], "video": final}
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg error: " + e.stderr.decode(errors="ignore")[-300:])