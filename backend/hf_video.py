"""
FREE realistic AI image-to-video (Google Flow / Veo maari: image -> short real video clip)
using public Hugging Face ZeroGPU Spaces through gradio_client.   Cost: Rs 0.

Reality check (free-la enna kedaikkum):
  ZeroGPU daily GPU quota  ->  anonymous 2 min | FREE HF account 5 min | PRO 40 min
  Oru clip ~ 60-90 sec GPU edukkum  =>  free account-la naalukku ~3-5 clips.
  Adhunaala: video-la ellaa scene-kum AI clip podaama, HF_MAX_AI_SCENES (default 3) scenes-ku mattum.
  Meedhi scenes-ku free 2.5D engine (anim3d.py) use aagum. Quota mudinja automatic-a adhukku maarum.

.env:
  HF_TOKEN=hf_xxx              <- IMPORTANT. huggingface.co/settings/tokens (free, "Read" token)
  HF_SPACES=multimodalart/wan2-1-fast,Lightricks/ltx-video-distilled     (default idhu dhaan)
  HF_MAX_AI_SCENES=3           (ovvoru video-kum max AI clips)
  HF_CLIP_SECONDS=3            (clip length; Space allow pannala-na 2 sec-ku retry)
  HF_TIMEOUT=420               (queue + generation max seconds)
  HF_ENABLE=off                (AI video-va muzhusa off panna)
  HF_H=384 HF_W=672            (wan Space resolution; multiples of 32)

Space maariyirukkalaam / sleep-la irukkalaam -> andha Space-a 10 min skip panni adutha Space try pannum.
Ellame fail aana exception -> pipeline free 2.5D engine-ku povum (video nikkaadhu).
"""
import os
import re
import shutil
import threading
import time

HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_SPACES = [s.strip() for s in os.getenv(
    "HF_SPACES", os.getenv("HF_SPACE", "") + ",multimodalart/wan2-1-fast,Lightricks/ltx-video-distilled"
).split(",") if s.strip()]
HF_SPACES = list(dict.fromkeys(HF_SPACES))                    # duplicates remove, order maintain
HF_API_NAME = os.getenv("HF_API_NAME", "").strip() or None    # unknown Space-ku mattum
HF_MAX_AI_SCENES = int(os.getenv("HF_MAX_AI_SCENES", "3"))
HF_CLIP_SECONDS = float(os.getenv("HF_CLIP_SECONDS", "3"))
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "420"))
HF_H = int(os.getenv("HF_H", "384"))
HF_W = int(os.getenv("HF_W", "672"))
ENABLED = os.getenv("HF_ENABLE", "on").lower() != "off" and bool(HF_SPACES)

NEG = "worst quality, blurry, jittery, distorted, deformed, extra limbs, text, watermark, static"

_lock = threading.Lock()          # oru neram-la oru GPU call (quota vaste aagaadhu)
_clients = {}
_space_off = {}                   # space -> time (andha varai skip)
_acct_off = [0.0]                 # quota mudinja ellaa Space-kum
_generic_api = {}
_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg)


def _have_client():
    try:
        import gradio_client  # noqa: F401
        return True
    except Exception:
        _warn_once("nogc", "[hf-video] gradio_client install aagala -> pip install gradio_client")
        return False


def available():
    return ENABLED and time.time() >= _acct_off[0] and _have_client() and \
        any(time.time() >= _space_off.get(s, 0) for s in HF_SPACES)


def pick_scenes(n):
    """N scenes-la evenly spread-a HF_MAX_AI_SCENES scenes-ku AI clip. eg n=5,k=3 -> {0,2,4}"""
    k = max(0, min(HF_MAX_AI_SCENES, n))
    if k == 0:
        return set()
    if k >= n:
        return set(range(n))
    return {round(i * (n - 1) / (k - 1)) if k > 1 else 0 for i in range(k)}


# ---------------------------------------------------------------- client
def _client(space):
    if space in _clients:
        return _clients[space]
    from gradio_client import Client
    c = None
    if HF_TOKEN:
        for key in ("token", "hf_token"):          # gradio_client versions: token (new) / hf_token (old)
            try:
                c = Client(space, verbose=False, **{key: HF_TOKEN})
                break
            except TypeError:
                continue
    if c is None:
        c = Client(space, verbose=False)
    _clients[space] = c
    return c


# ---------------------------------------------------------------- known Spaces (signature source code-la irundhu edutha)
def _kw_wan_fast(img, prompt, dur):
    # multimodalart/wan2-1-fast: generate_video(input_image, prompt, height, width, negative_prompt,
    #   duration_seconds=2, guidance_scale=1, steps=4, seed=42, randomize_seed=False)
    from gradio_client import handle_file
    return "/generate_video", dict(input_image=handle_file(img), prompt=prompt, height=HF_H, width=HF_W,
                                   negative_prompt=NEG, duration_seconds=dur, guidance_scale=1,
                                   steps=4, randomize_seed=True)


def _kw_ltx_distilled(img, prompt, dur):
    # Lightricks/ltx-video-distilled: image_to_video tab (api_name="image_to_video"). mode/height/width
    # Space-oda own defaults. (size="xlarge" => quota 2x, adhunaala 2nd choice)
    from gradio_client import handle_file
    return "/image_to_video", dict(prompt=prompt, negative_prompt=NEG, input_image_filepath=handle_file(img),
                                   duration_ui=dur, randomize_seed=True)


PROFILES = {
    "multimodalart/wan2-1-fast": _kw_wan_fast,
    "lightricks/ltx-video-distilled": _kw_ltx_distilled,
}


# ---------------------------------------------------------------- unknown Spaces (auto-detect)
def _default_for(p):
    if p.get("parameter_has_default"):
        return p.get("parameter_default")
    comp = str(p.get("component", "")).lower()
    if comp == "checkbox":
        return False
    if comp == "textbox":
        return ""
    if p.get("example_input") is not None:
        return p["example_input"]
    raise RuntimeError(f"parameter '{p.get('parameter_name')}' value theriyala")


def _generic_kwargs(space, img, prompt, dur):
    from gradio_client import handle_file
    c = _client(space)
    if space not in _generic_api:
        info = c.view_api(return_format="dict", print_info=False)
        best = None
        for name, ep in info.get("named_endpoints", {}).items():
            if HF_API_NAME and name != HF_API_NAME:
                continue
            params = ep.get("parameters", [])
            if any("image" in (str(p.get("parameter_name")) + str(p.get("component"))).lower() for p in params):
                score = (2 if "video" in name.lower() else 0) + (1 if "gen" in name.lower() else 0)
                if best is None or score > best[0]:
                    best = (score, name, params)
        if not best:
            raise RuntimeError("image-to-video endpoint kedaikala (HF_API_NAME set pannunga)")
        _generic_api[space] = best[1:]
    name, params = _generic_api[space]
    kw, used_img, used_txt = {}, False, False
    for p in params:
        pn = p["parameter_name"]
        low = (pn + str(p.get("label", ""))).lower()
        comp = str(p.get("component", "")).lower()
        if not used_img and ("image" in pn.lower() or comp in ("image", "imageeditor")):
            kw[pn] = handle_file(img)
            used_img = True
        elif comp == "textbox" and "neg" not in low:
            if not used_txt:
                kw[pn] = prompt
                used_txt = True
            else:
                kw[pn] = _default_for(p)
        elif "height" in low:
            kw[pn] = HF_H
        elif "width" in low:
            kw[pn] = HF_W
        elif "duration" in low:
            kw[pn] = dur
        else:
            kw[pn] = _default_for(p)
    return name, kw


# ---------------------------------------------------------------- run
def _find_video(obj):
    if isinstance(obj, str):
        return obj if obj.lower().endswith((".mp4", ".webm", ".mov")) else None
    if isinstance(obj, dict):
        for k in ("video", "path", "name", "value", "url"):
            if k in obj:
                v = _find_video(obj[k])
                if v:
                    return v
        return None
    if isinstance(obj, (list, tuple)):
        for it in obj:
            v = _find_video(it)
            if v:
                return v
    return None


def _quota_wait(msg):
    """'exceeded your GPU quota ... Try again in 0:12:34' -> seconds (illana 1 hour)."""
    m = re.search(r"(\d+):(\d{2}):(\d{2})", msg)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + 30
    return 3600


def _is_quota(msg):
    m = msg.lower()
    return ("quota" in m and ("gpu" in m or "zero" in m or "exceed" in m)) or ("zerogpu" in m and "limit" in m)


def _run_space(space, img, prompt, out):
    prof = PROFILES.get(space.lower())
    last = None
    for dur in dict.fromkeys([HF_CLIP_SECONDS, 2]):             # 3 sec work aagala-na 2 sec
        c = _client(space)
        name, kw = prof(img, prompt, dur) if prof else _generic_kwargs(space, img, prompt, dur)
        job = c.submit(api_name=name, **kw)
        try:
            res = job.result(timeout=HF_TIMEOUT)
        except Exception as e:
            try:
                job.cancel()
            except Exception:
                pass
            last = e
            msg = str(e).lower()
            if dur != 2 and any(w in msg for w in ("duration", "maximum", "minimum", "range", "validation")):
                continue
            raise
        video = _find_video(res)
        if not video or not os.path.exists(video):
            raise RuntimeError("Space video return pannala: " + str(res)[:200])
        shutil.copyfile(video, out)
        return
    raise last


def animate(img_path, prompt, out_path):
    """image -> short realistic mp4. Raises if every Space fails (pipeline free engine-ku povum)."""
    errors = []
    with _lock:
        for space in HF_SPACES:
            if time.time() < _space_off.get(space, 0):
                continue
            if time.time() < _acct_off[0]:
                break
            try:
                t0 = time.time()
                _run_space(space, img_path, prompt, out_path)
                print(f"[hf-video] {space} OK in {time.time() - t0:.0f}s")
                return
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                errors.append(f"{space}: {msg[:160]}")
                print(f"[hf-video] {space} fail: {msg[:300]}")
                if _is_quota(msg):
                    wait = _quota_wait(msg)
                    _acct_off[0] = time.time() + wait
                    print(f"[hf-video] GPU quota mudinjiduchu -> {wait // 60} min AI clips skip "
                          f"(HF_TOKEN set pannina quota athigam)")
                    break
                _clients.pop(space, None)
                _space_off[space] = time.time() + 600
    raise RuntimeError("; ".join(errors) or "HF AI video available illa")