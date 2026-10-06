"""
Channel-style rhyme videos (FREE, CPU only)

1. HERO mode  : ONE main 3D cartoon character (generated once, cut out with rembg) dances in front of
                a different background per scene. Characters ellaa scene-layum same-a irukkum.
2. SONG mode  : Neenga kudukkura paatu (mp3) -> Gemini kettu lyrics + time kandupidikkum ->
                ovvoru lyric part-ku match aagura background -> paatu thaan main audio.

anim3d.py (2.5D dance engine) reuse pannudhu.
"""
import os
import subprocess

from PIL import Image, ImageOps

W, H = 1280, 720
FPS = 25
MAX_SONG_SECONDS = int(os.getenv("MAX_SONG_SECONDS", "180"))   # 3 nimidam varai (CPU render time)
MAX_SONG_SCENES = int(os.getenv("MAX_SONG_SCENES", "14"))

HERO_STYLE = ("cute colorful kids cartoon, soft 3D cartoon render like popular nursery-rhyme videos, "
              "round chubby adorable character, huge shiny eyes, big happy smile, bright rainbow candy colors, "
              "no text, no watermark, ")
BG_STYLE = ("cute colorful kids cartoon, soft 3D cartoon render like popular nursery-rhyme videos, cheerful "
            "environment with clouds flowers and hills, wide scene with open ground in the foreground, "
            "NO characters, NO people, NO animals, bright rainbow candy colors, no text, no watermark, ")
GENERIC_BACKGROUNDS = [
    "sunny green meadow with colorful flowers", "bright garden with a rainbow in the sky",
    "starry night sky with a big glowing moon", "cozy forest with tall friendly trees",
    "blue sky with fluffy clouds over green hills", "playground with swings and a slide",
    "sandy beach with gentle waves", "magical candy land with lollipop trees",
]


# ---------------------------------------------------------------- hero character
def make_hero(character, workdir):
    """ONE main character sprite (RGBA, cropped) - illana None."""
    from . import pipeline as P
    prompt = (f"{character}, ONE character only, full body from head to toe, standing, "
              "facing the camera, plain solid light gray background, nothing else in the picture")
    for k in range(3):
        path = os.path.join(workdir, f"hero{k}.jpg")
        P.make_image(prompt, path, seed=777 + k * 13, style=HERO_STYLE)
        seg = P._segment(path)
        if not seg:
            continue
        _, fg, alpha = seg
        bbox = alpha.point(lambda p: 255 if p > 40 else 0).getbbox()
        if not bbox:
            continue
        sprite = fg.crop(bbox)
        if sprite.height >= 120:
            return sprite
    return None


def compose_scene(bg_path, sprite, idx):
    """Background + hero -> (im, alpha, bg) (anim3d.build_layers-ku thevaiyaana format)."""
    bg = Image.open(bg_path).convert("RGB").resize((W, H), Image.LANCZOS)
    sh = int(H * 0.62)
    sw = int(sprite.width * sh / sprite.height)
    if sw > W * 0.45:
        sw = int(W * 0.45)
        sh = int(sprite.height * sw / sprite.width)
    spr = sprite.resize((sw, sh), Image.LANCZOS)
    cx = (0.5, 0.3, 0.7, 0.5)[idx % 4] * W
    if cx > W / 2:
        spr = ImageOps.mirror(spr)                       # nadu pakkam paarkkum
    gy = int(H * 0.95)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(spr, (int(cx - sw / 2), gy - sh), spr)
    im = bg.copy()
    im.paste(canvas, (0, 0), canvas)
    return im, canvas.getchannel("A"), bg


def hero_layers(bg_path, sprite, idx):
    from . import anim3d
    im, alpha, bg = compose_scene(bg_path, sprite, idx)
    return anim3d.build_layers(im, alpha, bg, W, H)


# ---------------------------------------------------------------- song analysis
def prepare_song(song_path, workdir):
    from . import pipeline as P
    out = os.path.join(workdir, "song.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", song_path, "-vn", "-t", str(MAX_SONG_SECONDS),
                    "-ac", "2", "-ar", "44100", "-b:a", "160k", out], check=True, capture_output=True)
    return out, P.audio_duration(out)


def normalize_scenes(raw, duration):
    """Gemini timeline-a no-gap, no-overlap timeline-a maatthum."""
    sc = []
    for s in raw or []:
        try:
            st, en = float(s.get("start")), float(s.get("end"))
        except (TypeError, ValueError):
            continue
        sc.append({"start": st, "end": en, "lyric": str(s.get("lyric") or "").strip(),
                   "background_prompt": str(s.get("background_prompt") or "").strip()})
    sc = [s for s in sc if s["background_prompt"]]
    sc.sort(key=lambda s: s["start"])
    if not sc:
        return []
    sc[0]["start"] = 0.0
    for a, b in zip(sc, sc[1:]):
        a["end"] = b["start"]
    sc[-1]["end"] = duration
    out = []
    for s in sc:
        length = s["end"] - s["start"]
        if length <= 0:
            continue
        if out and length < 3.0:                          # romba short scene-a munnadi scene-oda serkkum
            out[-1]["end"] = s["end"]
            out[-1]["lyric"] = (out[-1]["lyric"] + " " + s["lyric"]).strip()
        else:
            out.append(s)
    while len(out) > MAX_SONG_SCENES:                     # athigam-na shortest-a merge pannum
        i = min(range(len(out)), key=lambda j: out[j]["end"] - out[j]["start"])
        j = i - 1 if i > 0 else i + 1
        a, b = sorted((i, j))
        out[a]["end"] = out[b]["end"]
        out[a]["lyric"] = (out[a]["lyric"] + " " + out[b]["lyric"]).strip()
        del out[b]
    return out


def fallback_scenes(duration, topic, lyrics):
    """Gemini illa/fail -> even split. Lyrics kuduthirundha lines-a pirichu kodukkum."""
    n = min(MAX_SONG_SCENES, max(4, round(duration / 10)))
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip()]
    out = []
    for k in range(n):
        chunk = lines[k * len(lines) // n:(k + 1) * len(lines) // n] if lines else []
        bg = GENERIC_BACKGROUNDS[k % len(GENERIC_BACKGROUNDS)]
        out.append({"start": duration * k / n, "end": duration * (k + 1) / n,
                    "lyric": "\n".join(chunk),
                    "background_prompt": f"{topic}, {bg}" if topic else bg})
    return out


def analyze_song(mp3, workdir, topic, lyrics, duration):
    """Returns {"title", "character", "scenes":[{start,end,lyric,background_prompt}]}. Eppavum work aagum."""
    from . import pipeline as P
    default = {"title": topic or "Kids Song",
               "character": (f"a cute cartoon animal friend for a kids song about {topic}, bright colors"
                             if topic else "a cute cartoon bear cub wearing a red shirt"),
               "scenes": None}
    if P.GEMINI_API_KEY:
        try:
            small = os.path.join(workdir, "song_small.mp3")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3, "-ac", "1", "-ar", "16000",
                            "-b:a", "32k", small], check=True, capture_output=True)
            n = min(MAX_SONG_SCENES, max(4, round(duration / 10)))
            prompt = (
                f"Listen to this children's song ({duration:.0f} seconds long). Return ONLY JSON:\n"
                '{"title": "...", "character": "...", "scenes": [{"start": 0.0, "end": 9.5, '
                '"lyric": "...", "background_prompt": "..."}]}\n\n'
                "Rules:\n"
                f"- scenes must cover the WHOLE song in order with no gaps, about {n} scenes, each at least "
                "4 seconds. start/end are seconds (numbers) in the audio.\n"
                "- Split at natural lyric line / verse boundaries. lyric = the exact words sung in that part, "
                "in the original language and script (empty string for instrumental parts).\n"
                "- character: ONE cute main cartoon character (animal or child) that fits the song, in "
                "English, one sentence (species, colors, clothes).\n"
                "- background_prompt: ENGLISH, the place/setting matching the lyrics of that scene (things "
                "mentioned in the lyrics, e.g. moon, stars, rain, fruits). Absolutely no characters, "
                "people or animals in it.\n- Safe for kids.\n")
            if topic:
                prompt += f"Song topic hint: {topic}\n"
            if (lyrics or "").strip():
                prompt += f"\nLyrics provided by the user (use these words, align them to the audio):\n{lyrics.strip()}\n"
            data = P.extract_json(P.gemini_generate(prompt, audio_path=small))
            scenes = normalize_scenes(data.get("scenes"), duration)
            if len(scenes) >= 2:
                return {"title": data.get("title") or default["title"],
                        "character": data.get("character") or default["character"], "scenes": scenes}
        except Exception as e:
            print(f"[song] Gemini analysis fail: {e}")
    default["scenes"] = fallback_scenes(duration, topic, lyrics)
    return default


# ---------------------------------------------------------------- song video
def run_song(topic, song_path, lyrics, use_subs, workdir, on_progress):
    import numpy as np
    from . import anim3d, pipeline as P
    try:
        on_progress(0.03, "Paatu-a kekkuren (lyrics + timing)...")
        mp3, dur = prepare_song(song_path, workdir)
        if dur < 5:
            raise ValueError("Paatu romba short. Kuraindhadhu 5 second venum.")
        info = analyze_song(mp3, workdir, topic, lyrics, dur)

        on_progress(0.1, "Main character create panren...")
        hero = None
        if P.ANIM3D:
            try:
                hero = make_hero(info["character"], workdir)
            except Exception as e:
                print(f"[song] hero fail: {e}")

        total = int(round(dur * FPS))
        env = P.audio_envelope(mp3, total)
        scenes, clips = info["scenes"], []
        for k, sc in enumerate(scenes):
            on_progress(0.12 + 0.76 * k / len(scenes),
                        f"Scene {k + 1}/{len(scenes)}: lyrics-ku match-a background + dance...")
            f0 = int(round(sc["start"] * FPS))
            f1 = total if k == len(scenes) - 1 else int(round(sc["end"] * FPS))
            frames = max(1, f1 - f0)
            env_k = env[f0:f1]
            if len(env_k) < frames:
                env_k = np.pad(env_k, (0, frames - len(env_k)))
            seg = os.path.join(workdir, f"seg{k}.mp3")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{f0 / FPS:.3f}", "-t", f"{frames / FPS:.3f}",
                            "-i", mp3, seg], check=True, capture_output=True)
            img = os.path.join(workdir, f"img{k}.jpg")
            clip = os.path.join(workdir, f"clip{k}.mp4")
            sub = sc["lyric"] if (use_subs and sc["lyric"]) else None
            clip_dur = frames / FPS
            done = False
            if hero is not None:
                try:
                    P.make_image(sc["background_prompt"], img, seed=2000 + k, style=BG_STYLE)
                    layers = hero_layers(img, hero, k)
                    P._encode_frames(lambda: anim3d.render_frames(layers, env_k, frames, k, FPS, True),
                                     seg, clip_dur, clip, workdir, k, sub)
                    done = True
                except Exception as e:
                    print(f"[song] scene {k + 1} hero fail: {e}")
            if not done:   # hero illa -> oru image-la character + background; 3D engine illa-na camera move
                P.make_image(f"{info['character']}, {sc['background_prompt']}", img, seed=2000 + k)
                try:
                    if not P.ANIM3D:
                        raise RuntimeError("anim3d off")
                    layers = P.extract_layers3d(img)
                    if not layers:
                        raise RuntimeError("character kedaikala")
                    P._encode_frames(lambda: anim3d.render_frames(layers, env_k, frames, k, FPS, True),
                                     seg, clip_dur, clip, workdir, k, sub)
                except Exception as e:
                    print(f"[song] scene {k + 1} 3D fail ({e}) -> camera move")
                    P.make_clip(img, seg, clip, workdir, k, sub, pad=0.0)
            clips.append(clip)

        on_progress(0.9, "Scenes-a join panren...")
        joined = os.path.join(workdir, "joined.mp4")
        P.join_clips(clips, joined, workdir)
        on_progress(0.95, "Original paatu-a serkuren...")
        final = os.path.join(workdir, "song_video.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", mp3, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", final],
                       check=True, capture_output=True)
        P._usage["count"] += 1
        return {"title": info["title"], "scenes": [s["lyric"] or "♪" for s in scenes], "video": final}
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg error: " + (e.stderr or b"").decode(errors="ignore")[-300:])