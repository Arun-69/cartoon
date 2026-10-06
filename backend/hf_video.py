"""
OPTIONAL free AI image-to-video using a public Hugging Face Space (gradio_client).
Cost: Rs 0. Limits: ZeroGPU free quota (daily), queue wait, Space-a maathitta API maaralaam.

Enable:   HF_SPACE=owner/space-name      (huggingface.co/spaces -> search "image to video")
Optional: HF_TOKEN=hf_xxx   (free HF account token -> konjam athiga free GPU quota)
          HF_API_NAME=/generate_video  (auto-detect work aagala-na mattum)

Fail aana (quota mudinjiduchu / Space sleep / error) -> exception -> pipeline automatic-a
free 3D engine-ku (anim3d.py) poidum. Quota error vandha 30 nimidam HF-a skip pannum.
"""
import os
import shutil
import time

HF_SPACE = os.getenv("HF_SPACE", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_API_NAME = os.getenv("HF_API_NAME", "").strip() or None
ENABLED = bool(HF_SPACE)

_state = {"client": None, "api": None, "off_until": 0.0}


def available():
    return ENABLED and time.time() >= _state["off_until"]


def _client():
    if _state["client"] is None:
        from gradio_client import Client
        _state["client"] = Client(HF_SPACE, hf_token=HF_TOKEN, verbose=False)
    return _state["client"]


def _pick_endpoint(client):
    """Image + prompt eduthukkura endpoint-a kandupidikkum."""
    info = client.view_api(return_format="dict", print_info=False)
    eps = info.get("named_endpoints", {})
    best = None
    for name, ep in eps.items():
        if HF_API_NAME and name != HF_API_NAME:
            continue
        params = ep.get("parameters", [])
        img = [p for p in params if str(p.get("component", "")).lower() in ("image", "imageeditor")
               or "image" in str(p.get("parameter_name", "")).lower()]
        txt = [p for p in params if str(p.get("component", "")).lower() == "textbox"
               and "neg" not in (str(p.get("parameter_name", "")) + str(p.get("label", ""))).lower()]
        if not img:
            continue
        score = (2 if "video" in name.lower() else 0) + (1 if "generate" in name.lower() else 0) \
            + (1 if txt else 0)
        if best is None or score > best[0]:
            best = (score, name, params, img[0], txt[0] if txt else None)
    if not best:
        raise RuntimeError("HF Space-la image-to-video endpoint kedaikala. HF_API_NAME set pannunga.")
    return best[1:]


def _default_for(p):
    if p.get("parameter_has_default"):
        return p.get("parameter_default")
    comp = str(p.get("component", "")).lower()
    if comp == "checkbox":
        return False
    if comp == "textbox":
        return ""
    ex = p.get("example_input")
    if ex is not None:
        return ex
    raise RuntimeError(f"HF Space parameter '{p.get('parameter_name')}' value theriyala.")


def _find_video(obj):
    """Result-la irundhu video file path-a edukkum (str / dict / list / tuple)."""
    if isinstance(obj, str):
        return obj if obj.lower().endswith((".mp4", ".webm", ".mov")) else None
    if isinstance(obj, dict):
        for k in ("video", "path", "name", "value"):
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


def animate(img_path, prompt, out_path):
    """img -> short animated mp4 (free HF Space). Raises on any failure."""
    from gradio_client import handle_file
    try:
        client = _client()
        if _state["api"] is None:
            _state["api"] = _pick_endpoint(client)
        name, params, img_p, txt_p = _state["api"]
        kwargs = {}
        for p in params:
            pn = p["parameter_name"]
            if p is img_p:
                kwargs[pn] = handle_file(img_path)
            elif txt_p is not None and p is txt_p:
                kwargs[pn] = prompt
            else:
                kwargs[pn] = _default_for(p)
        result = client.predict(api_name=name, **kwargs)
        video = _find_video(result)
        if not video or not os.path.exists(video):
            raise RuntimeError("HF Space video return pannala: " + str(result)[:200])
        shutil.copyfile(video, out_path)
    except Exception as e:
        msg = str(e).lower()
        if any(w in msg for w in ("quota", "gpu", "zerogpu", "rate", "limit", "exceeded")):
            _state["off_until"] = time.time() + 30 * 60      # quota mudinjiduchu -> 30 min skip
        raise