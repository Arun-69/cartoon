"""
FREE pseudo-3D character animation (CPU only, no paid API, no GPU).

Idea (2.5D):
  1. Image-a rendu layer-a pirikkirom: background (character irundha idam nirappi) + character cutout.
  2. Character-ku "inflated" depth map (nadu = nearer, ovvaram = farther) -> camera orbit panna
     character 3D-a thiruppura maari theriyum (parallax). Background-ku ground-plane depth.
  3. Character-ku squash & stretch hop (dance), body lean, jelly sway - voice loudness-oda sync.
  4. Ground shadow (hop panna chinnadhaagum), bokeh circles (back), sparkles + bubbles (front).

Public API:
    layers = build_layers(im_rgb, alpha, bg_rgb, W, H)
    for raw_rgb_bytes in render_frames(layers, env, frames, idx, fps, rhyme): ...
"""
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .cartoon_audio import beat_rate

PASTEL = [(255, 255, 255), (255, 236, 140), (255, 170, 200), (170, 225, 255), (190, 255, 200)]
K_CHAR = 46.0     # character parallax strength (pixels)
K_BG = 70.0       # background parallax strength (pixels)


# ---------------------------------------------------------------- layers
def _make_particles(W, H, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(34):
        front = i >= 14                       # first 14 = bokeh (back), rest = sparkles/bubbles (front)
        parts.append(dict(
            front=front,
            kind=("bokeh" if not front else ("star" if i % 3 else "bubble")),
            x=float(rng.uniform(0, W)), y=float(rng.uniform(0, H)),
            speed=float(rng.uniform(10, 26) if not front else rng.uniform(22, 55)),
            size=float(rng.uniform(22, 52) if not front else rng.uniform(7, 17)),
            phase=float(rng.uniform(0, 6.28)),
            color=PASTEL[int(rng.integers(0, len(PASTEL)))],
        ))
    return parts


def build_layers(im, alpha, bg, W, H, seed=0, realistic=False):
    """im: PIL RGB (WxH), alpha: PIL 'L' mask, bg: PIL RGB background (character removed)."""
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    a8 = np.asarray(alpha.convert("L"), dtype=np.uint8)
    # edge-a konjam soft aakkuvom (cut halo theriyaadhu)
    a_f = cv2.GaussianBlur(a8, (0, 0), 0.8).astype(np.float32) / 255.0
    binary = a8 > 90
    # rembg-la varra chinna speckle/debris blobs-a remove pannu (anchor, shadow, depth thappa aagaadhu)
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n_cc > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= max(areas.max() * 0.04, 200))[0]
        binary = np.isin(lab, keep)
        a_f = a_f * cv2.dilate(binary.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    ys, xs = np.nonzero(binary)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    anchor = ((x0 + x1) / 2.0, float(y1))                 # paadham (feet)
    ch = float(max(40, y1 - y0))                          # character height (px)
    bw = float(max(40, x1 - x0))

    # inflated depth: nadu nearer (1.0), ovvaram farther (0.55)
    dist = distance_transform_edt(binary).astype(np.float32)
    bulge = np.sqrt(np.clip(dist / (dist.max() + 1e-6), 0, 1))
    depth_c = (0.55 + 0.45 * bulge).astype(np.float32)

    # premultiplied RGBA (4 channel uint8) -> dark fringe varaadhu
    pm = np.dstack([rgb.astype(np.float32) * a_f[..., None], a_f * 255.0])
    pm = np.clip(pm, 0, 255).astype(np.uint8)

    # background ground-plane depth (mela = far, keezha = near)
    ground = (0.05 + 0.40 * (np.arange(H, dtype=np.float32) / H) ** 1.2)[:, None]

    # ground shadow mask
    shadow = np.zeros((H, W), np.float32)
    cv2.ellipse(shadow, (int(anchor[0]), int(max(0, anchor[1] - bw * 0.02))),
                (int(bw * 0.38) + 1, int(max(6, bw * 0.07))), 0, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(4.0, bw * 0.04))

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return dict(W=W, H=H, bg=np.asarray(bg.convert("RGB"), dtype=np.uint8), pm=pm,
                depth_c=depth_c, anchor=anchor, ch=ch, ground=ground, shadow=shadow,
                X=X, Y=Y, parts=_make_particles(W, H, seed), realistic=realistic)


# ---------------------------------------------------------------- particles
def _star_pts(cx, cy, r):
    pts = []
    for k in range(8):
        ang = math.pi / 4 * k - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return np.array(pts, np.int32)


def _draw_particles(frame, parts, t, front, W, H):
    ov = frame.copy()
    drew = False
    for p in parts:
        if p["front"] != front:
            continue
        y = (p["y"] - p["speed"] * t) % (H + 120) - 60
        x = p["x"] + 22 * math.sin(p["phase"] + t * 0.8)
        r = p["size"]
        col = p["color"]
        if p["kind"] == "bokeh":
            cv2.circle(ov, (int(x), int(y)), int(r), col, -1, cv2.LINE_AA)
        elif p["kind"] == "star":
            tw = 0.55 + 0.45 * math.sin(p["phase"] * 2 + t * 5.0)
            cv2.fillPoly(ov, [_star_pts(x, y, r * (0.5 + tw))], col, cv2.LINE_AA)
        else:  # bubble
            cv2.circle(ov, (int(x), int(y)), int(r), (235, 248, 255), 2, cv2.LINE_AA)
            cv2.circle(ov, (int(x - r * 0.35), int(y - r * 0.35)), max(1, int(r * 0.2)),
                       (255, 255, 255), -1, cv2.LINE_AA)
        drew = True
    if drew:
        a = 0.30 if not front else 0.85
        cv2.addWeighted(ov, a, frame, 1 - a, 0, dst=frame)


# ---------------------------------------------------------------- frames
def render_frames(L, env, frames, idx, fps, rhyme):
    W, H = L["W"], L["H"]
    bg, pm, depth_c = L["bg"], L["pm"], L["depth_c"]
    ax, ay = L["anchor"]
    ch, ground, shadow = L["ch"], L["ground"], L["shadow"]
    X, Y = L["X"], L["Y"]
    parts = L["parts"]
    real = bool(L.get("realistic"))                    # realistic = hop/sparkles illa, subtle breathing + parallax

    env = np.convolve(np.asarray(env, np.float32), np.ones(5) / 5, mode="same")
    beat = beat_rate(rhyme)                            # hops per second (cartoon_audio-oda sync)
    zoom_in = idx % 2 == 0
    height_row = np.clip((ay - np.arange(H, dtype=np.float32)) / ch, 0, 1.2)[:, None]
    jelly_shape = height_row ** 1.6                    # mela pogap pogap athigam asaiyum

    for n in range(frames):
        t = n / fps
        e = float(env[min(n, len(env) - 1)])
        amp = 0.30 + 0.70 * e                          # silent-la konjam, pesumbodhu athigam
        ph = t * beat

        hop = abs(math.sin(math.pi * ph))              # 0 = nilathula, 1 = ucchi
        jump = ch * (0.045 if rhyme else 0.030) * amp * hop
        contact = (1.0 - hop) ** 3
        squash = 0.05 * amp * contact
        stretch = 0.03 * amp * hop
        breathe = 0.008 * math.sin(2 * math.pi * 0.5 * t)
        sy = 1.0 - squash + stretch + breathe + 0.02 * e
        sx = 1.0 / math.sqrt(sy)                       # volume maintain
        theta = math.radians(3.0) * amp * math.sin(math.pi * ph + idx * 0.7)
        dx = 8.0 * math.sin(2 * math.pi * 0.3 * t + idx)
        jx = ch * 0.025 * amp * math.sin(2 * math.pi * 0.5 * beat * t + 0.8) * jelly_shape
        yaw = math.sin(2 * math.pi * 0.18 * t + idx * 1.3)          # camera orbit -1..1
        if real:                                       # cinematic: subtle breathing + head-sway, no cartoon bounce
            hop = 0.0
            jump = 0.0
            sy = 1.0 + 0.007 * math.sin(2 * math.pi * 0.33 * t + idx) + 0.004 * e
            sx = 1.0 / math.sqrt(sy)
            theta = math.radians(0.8) * math.sin(2 * math.pi * 0.22 * t + idx * 0.7)
            dx = 4.0 * math.sin(2 * math.pi * 0.17 * t + idx)
            jx = ch * 0.004 * (0.4 + e) * math.sin(2 * math.pi * 0.6 * t + 0.8) * jelly_shape
            yaw = 1.0 * math.sin(2 * math.pi * 0.12 * t + idx * 1.3)    # slower camera orbit (smear kuraiva)

        # ---- background (ground-plane parallax)
        Xb = (X - yaw * K_BG * (ground - 0.5)).astype(np.float32)
        frame_bg = cv2.remap(bg, Xb, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if not real:
            _draw_particles(frame_bg, parts, t, False, W, H)

        # ---- ground shadow (hop panna light-aagum)
        k = 0.40 * (1.0 - 0.55 * hop)
        bgf = frame_bg.astype(np.float32) * (1.0 - k * shadow[..., None])

        # ---- character: lean + squash/stretch + hop + jelly, apprm depth parallax
        relx = X - ax - dx - jx
        rely = Y - ay + jump
        c, s = math.cos(theta), math.sin(theta)
        Xs = (ax + (c * relx + s * rely) / sx).astype(np.float32)
        Ys = (ay + (-s * relx + c * rely) / sy).astype(np.float32)
        d_s = cv2.remap(depth_c, Xs, Ys, cv2.INTER_LINEAR, borderValue=0.5)
        Xs2 = (Xs - yaw * K_CHAR * (d_s - 0.5)).astype(np.float32)
        chr_pm = cv2.remap(pm, Xs2, Ys, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = chr_pm[..., 3:4].astype(np.float32) / 255.0
        out = bgf * (1.0 - a) + chr_pm[..., :3].astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)

        if not real:
            _draw_particles(out, parts, t, True, W, H)

        # ---- slow camera push / pull
        kk = n / max(1, frames - 1)
        z = 1.0 + 0.06 * (kk if zoom_in else 1.0 - kk)
        M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]], np.float32)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        yield out.tobytes()


# ---------------------------------------------------------------- intro / outro card
def card_background(W, H):
    """Sky gradient + rainbow + clouds + hills (title card background)."""
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top = np.array([110, 190, 255], np.float32)
    bot = np.array([255, 232, 246], np.float32)
    bg = np.broadcast_to(top * (1 - ys) + bot * ys, (H, W, 3)).astype(np.uint8).copy()
    cx, cy = W // 2, int(H * 0.93)
    cols = [(255, 80, 80), (255, 160, 60), (255, 230, 70), (90, 210, 110), (80, 170, 255), (150, 100, 230)]
    for i, c in enumerate(cols):
        cv2.ellipse(bg, (cx, cy), (int(W * 0.44) - i * 22, int(H * 0.50) - i * 22), 0, 180, 360, c, 24, cv2.LINE_AA)

    def cloud(x, y, sz):
        for dx, dy, r in [(-1.0, 0.2, .6), (-.4, -.2, .8), (.3, -.1, .9), (1.0, .2, .6), (0, .3, 1.0)]:
            cv2.circle(bg, (int(x + dx * sz), int(y + dy * sz)), int(r * sz * 0.8), (255, 255, 255), -1, cv2.LINE_AA)

    cloud(cx - W * 0.44, cy - 5, 70)
    cloud(cx + W * 0.44, cy - 5, 70)
    cloud(W * 0.22, H * 0.32, 45)
    cloud(W * 0.80, H * 0.36, 55)
    cv2.ellipse(bg, (W // 4, H + 70), (W // 2, 170), 0, 0, 360, (120, 205, 120), -1, cv2.LINE_AA)
    cv2.ellipse(bg, (3 * W // 4, H + 90), (W // 2, 190), 0, 0, 360, (100, 190, 110), -1, cv2.LINE_AA)
    return bg


def render_card_frames(W, H, frames, fps, seed=0):
    bg = card_background(W, H)
    parts = _make_particles(W, H, seed)
    for n in range(frames):
        t = n / fps
        fr = bg.copy()
        # rotating sun (top-left)
        sx, sy = int(W * 0.09), int(H * 0.16)
        for k in range(12):
            ang = t * 0.7 + k * math.pi / 6
            cv2.line(fr, (int(sx + 62 * math.cos(ang)), int(sy + 62 * math.sin(ang))),
                     (int(sx + 92 * math.cos(ang)), int(sy + 92 * math.sin(ang))), (255, 220, 60), 6, cv2.LINE_AA)
        cv2.circle(fr, (sx, sy), 52, (255, 232, 90), -1, cv2.LINE_AA)
        _draw_particles(fr, parts, t, False, W, H)
        _draw_particles(fr, parts, t, True, W, H)
        yield fr.tobytes()