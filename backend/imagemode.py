"""
IMAGE mode: user upload panna image(s) -> appadiye animated video (FREE, CPU).

- Storyboard / comic-grid image-a (eg: 2x3 panels) automatic-a panels-aa pirichu, order-padi scenes-aa maatthum.
  Panel number badges (1,2,3...) irundha remove pannum (opencv irundha).
- Ovvoru image-um 16:9-ku fit aagum (pakkathula blur background, image stretch aagaadhu).
- Gemini image-a paathu ovvoru scene-kum narration ezhudhum (key illa-na silent video).
- Apram ungal ippo irukkura animation chain (HF free video -> anim3d 2.5D -> camera move) run aagum.
"""
import os

from PIL import Image, ImageDraw, ImageFilter

W, H = 1280, 720
MAX_UPLOAD_SCENES = int(os.getenv("MAX_UPLOAD_SCENES", "10"))


# ---------------------------------------------------------------- storyboard split
def _runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def _segments(gutter_mask, length, min_frac=0.12):
    """Gutter-kku naduvula irukkura content segments."""
    segs, prev = [], 0
    for a, b in _runs(gutter_mask) + [(length, length)]:
        if a - prev >= length * min_frac:
            segs.append((prev, a))
        prev = b
    return segs


def _strip_badge(panel):
    """Panel top-left-la irukkura number circle-a (1,2,3..) inpaint pannum."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return panel
    a = np.asarray(panel)
    h, w = a.shape[:2]
    r = int(min(h, w) * 0.24)
    reg = a[:r, :r]
    white = (reg.min(axis=2) > 235).astype("uint8")
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 500 <= area <= 6000 and 0.75 <= bw / max(bh, 1) <= 1.3 and x < r * 0.55 and y < r * 0.55:
            mask = np.zeros((h, w), np.uint8)
            cx, cy = x + bw // 2, y + bh // 2
            cv2.ellipse(mask, (cx, cy), (bw // 2 + 7, bh // 2 + 7), 0, 0, 360, 255, -1)
            return Image.fromarray(cv2.inpaint(a, mask, 6, cv2.INPAINT_TELEA))
    return panel


def split_storyboard(path):
    """Grid layout (white gutters) irundha panels list (reading order); illa-na None."""
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    h, w, _ = a.shape
    if min(h, w) < 300:
        return None
    corners = np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3),
                              a[-8:, :8].reshape(-1, 3), a[-8:, -8:].reshape(-1, 3)])
    paper = np.median(corners, axis=0)
    is_paper = np.abs(a - paper).max(axis=2) < 14
    cols = _segments(is_paper.mean(axis=0) > 0.985, w)
    rows = _segments(is_paper.mean(axis=1) > 0.985, h)
    if len(cols) * len(rows) < 2:
        return None
    panels = []
    for (y0, y1) in rows:
        for (x0, x1) in cols:
            p = im.crop((x0 + 5, y0 + 5, x1 - 5, y1 - 5))        # dark border-a vetti
            if p.width < 120 or p.height < 120:
                return None
            panels.append(_strip_badge(p))
    return panels


# ---------------------------------------------------------------- fit to 16:9
def fit_16x9(im):
    iw, ih = im.size
    if abs(iw / ih - W / H) < 0.03:
        return im.resize((W, H), Image.LANCZOS)
    s = max(W / iw, H / ih)
    cover = im.resize((int(iw * s) + 1, int(ih * s) + 1), Image.LANCZOS)
    x0, y0 = (cover.width - W) // 2, (cover.height - H) // 2
    bg = cover.crop((x0, y0, x0 + W, y0 + H)).filter(ImageFilter.GaussianBlur(60))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), 0.40)
    s2 = min(W / iw, H / ih)
    fg = im.resize((max(1, int(iw * s2)), max(1, int(ih * s2))), Image.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    return bg


def prepare_images(files, workdir):
    """Uploaded files -> 16:9 scene images (storyboards split aagum). Returns paths."""
    out = []
    for k, f in enumerate(files):
        try:
            panels = split_storyboard(f)
        except Exception as e:
            print(f"[images] storyboard split fail ({e!r}) -> oru image-a edukkuren")
            panels = None
        imgs = panels or [Image.open(f).convert("RGB")]
        for j, im in enumerate(imgs):
            p = os.path.join(workdir, f"up_{k}_{j}.jpg")
            fit_16x9(im).save(p, "JPEG", quality=92)
            out.append(p)
    return out[:MAX_UPLOAD_SCENES]


# ---------------------------------------------------------------- narration (Gemini vision)
def narrate(paths, topic, language, mode):
    """Returns (title, scenes) - scenes length == len(paths). Key illa-na narration "" (silent)."""
    from . import pipeline as P
    n = len(paths)
    scenes = [{"narration": "", "motion_prompt": "characters move happily, cartoon style",
               "image_prompt": "cartoon scene"} for _ in range(n)]
    title = (topic or "").strip()[:60] or "My Cartoon"
    if not P.GEMINI_API_KEY:
        print("[images] GEMINI_API_KEY illa -> narration illama (silent) video varum")
        return title, scenes
    try:
        small = []
        for k, p in enumerate(paths):
            sp = p.replace(".jpg", "_s.jpg")
            Image.open(p).resize((768, 432), Image.LANCZOS).save(sp, "JPEG", quality=85)
            small.append(sp)
        style = ("Narration must be RHYMING couplets (nursery-rhyme feel)." if mode.startswith("Rhyme")
                 else "Narration should be like a cheerful storyteller.")
        prompt = (
            f"You are given {n} images in order. They are consecutive scenes of a kids' cartoon. For each "
            f"image write: narration = 1-2 short simple cheerful sentences in {language} telling what is "
            "happening in that scene (kid friendly, no violence descriptions - keep chases playful), and "
            "motion_prompt = one short ENGLISH sentence about how the characters move. Keep the story "
            f"continuous across scenes. {style}\n"
            'Return ONLY JSON: {"title": "...", "scenes": [{"narration": "...", "motion_prompt": "..."}]} '
            f"with EXACTLY {n} scenes in the same order as the images.")
        if topic and topic.strip():
            prompt += f"\nStory hint from the user: {topic.strip()}"
        data = P.extract_json(P.gemini_generate(prompt, image_paths=small))
        got = data.get("scenes") or []
        for i in range(min(n, len(got))):
            nar = str(got[i].get("narration") or "").strip()
            scenes[i]["narration"] = nar
            scenes[i]["motion_prompt"] = str(got[i].get("motion_prompt") or scenes[i]["motion_prompt"])
            scenes[i]["image_prompt"] = nar or scenes[i]["image_prompt"]
        title = data.get("title") or title
    except Exception as e:
        print(f"[images] Gemini vision fail: {e}")
    return title, scenes