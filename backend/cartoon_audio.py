"""
Kids-cartoon audio, 100% synthesized (numpy only) -> free + copyright-safe.

  make_cartoon_tune(path)               bouncy marimba tune + bass + kick/clap/shaker (seamless loop)
  make_scene_sfx(path, dur, rhyme, ...) "boing" on every landing (hop-sync with anim3d) + sparkle "ting"
  make_jingle(path, dur, kind)          intro ("intro") / outro ("outro") jingle
"""
import wave

import numpy as np

SR = 44100
BEAT_RHYME = 1.7     # hops per second (anim3d-oda same)
BEAT_STORY = 0.9


def beat_rate(rhyme):
    return BEAT_RHYME if rhyme else BEAT_STORY


# ---------------------------------------------------------------- helpers
def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi(name):
    return 12 * (int(name[1:]) + 1) + _BASE[name[0]]


def _t(dur):
    return np.arange(int(dur * SR), dtype=np.float64) / SR


def _add(buf, start, sig, gain=1.0):
    i = int(start * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += gain * sig[: j - i]


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- instruments
def marimba(f, dur=0.5):
    t = _t(dur)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-6.5 * t)
         + 0.35 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-18 * t)
         + 0.15 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-9 * t))
    return s * np.minimum(1, t / 0.002)


def bell(f, dur=0.8):
    t = _t(dur)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-5 * t)
         + 0.5 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-8 * t)
         + 0.25 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-12 * t))
    return s * np.minimum(1, t / 0.002)


def bass(f, dur=0.34):
    t = _t(dur)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-5 * t)


def stab(freqs, dur=0.16):
    t = _t(dur)
    s = sum(np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t) for f in freqs)
    return s / len(freqs) * np.exp(-18 * t)


def kick(dur=0.25):
    t = _t(dur)
    f = 50 + 110 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)


def clap(rng, dur=0.12):
    t = _t(dur)
    return rng.standard_normal(len(t)) * np.exp(-t * 28)


def shaker(rng, dur=0.07):
    t = _t(dur)
    n = np.diff(rng.standard_normal(len(t) + 1))        # crude high-pass
    return n * np.exp(-t * 45)


def boing(dur=0.38):
    t = _t(dur)
    f = 150 + 300 * (1 - np.exp(-t * 30)) + 45 * np.sin(2 * np.pi * 30 * t) * np.exp(-t * 10)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    return s * np.minimum(1, t / 0.004)


# ---------------------------------------------------------------- background tune
def make_cartoon_tune(path, bpm=112):
    rng = np.random.default_rng(7)
    beat = 60.0 / bpm
    step = beat / 2                        # eighth note
    bars = 8
    total = bars * 4 * beat
    buf = np.zeros(int((total + 1.2) * SR))

    chords = [  # (bass midi, chord midi)
        (48, [60, 64, 67]), (45, [57, 60, 64]), (41, [53, 57, 60]), (43, [55, 59, 62]),
        (48, [60, 64, 67]), (45, [57, 60, 64]), (41, [53, 57, 60]), (43, [55, 59, 62]),
    ]
    mel = [
        "E5 E5 G5 E5 C5 E5 G5 -", "E5 E5 A5 E5 C5 E5 A5 -",
        "F5 F5 A5 F5 C5 F5 A5 -", "D5 D5 G5 D5 B4 D5 G5 -",
        "C5 E5 G5 C6 G5 E5 G5 -", "A4 C5 E5 A5 E5 C5 E5 -",
        "F5 A5 C6 A5 F5 A5 G5 F5", "D5 G5 B5 D6 B5 G5 C6 -",
    ]
    for b in range(bars):
        t0 = b * 4 * beat
        root, ch = chords[b]
        for k, n in enumerate(mel[b].split()):
            if n != "-":
                _add(buf, t0 + k * step, marimba(hz(midi(n)), 0.45), 0.50)
        for beat_i in range(4):
            tb = t0 + beat_i * beat
            if beat_i in (0, 2):                                   # oom
                _add(buf, tb, bass(hz(root)), 0.45)
                _add(buf, tb, kick(), 0.55)
            else:                                                  # pah
                _add(buf, tb, stab([hz(m) for m in ch]), 0.20)
                _add(buf, tb, clap(rng), 0.16)
            _add(buf, tb, shaker(rng), 0.05)
            _add(buf, tb + step, shaker(rng), 0.09)
    n = int(total * SR)
    tail = buf[n:]
    buf = buf[:n]
    buf[: len(tail)] += tail[: len(buf)]                           # seamless loop
    buf /= max(1e-6, np.abs(buf).max()) / 0.85
    write_wav(path, buf)


# ---------------------------------------------------------------- per-scene SFX (hop sync)
def make_scene_sfx(path, dur, rhyme, idx, env, fps):
    """env = voice envelope (per video frame, 0..1). Boing = exactly when the character lands."""
    rng = np.random.default_rng(100 + idx)
    buf = np.zeros(int(dur * SR))
    beat = beat_rate(rhyme)
    k = 1
    while k / beat < dur - 0.2:
        tk = k / beat
        if rhyme or k % 2 == 0:
            e = float(env[min(int(tk * fps), len(env) - 1)]) if len(env) else 0.5
            g = (0.55 + 0.45 * (0.3 + 0.7 * e)) * (0.30 if rhyme else 0.20)
            _add(buf, tk, boing(), g)
        k += 1
    t = float(rng.uniform(0.4, 1.0))
    while t < dur - 0.5:                                           # sparkle "ting"
        note = int(rng.choice([96, 100, 103, 108]))
        _add(buf, t, bell(hz(note), 0.6), 0.10)
        t += float(rng.uniform(1.0, 1.9))
    write_wav(path, buf)


# ---------------------------------------------------------------- intro / outro jingles
def make_jingle(path, dur, kind="intro"):
    buf = np.zeros(int(dur * SR))
    if kind == "intro":
        seq = ["C5", "E5", "G5", "C6"]
        for i, n in enumerate(seq):
            _add(buf, 0.15 + i * 0.13, marimba(hz(midi(n)), 0.5), 0.5)
        for n in ("C5", "E5", "G5", "C6"):
            _add(buf, 0.75, bell(hz(midi(n)), 1.4), 0.22)
        _add(buf, 0.75, kick(), 0.5)
        for j in range(5):
            _add(buf, 1.0 + j * 0.28, bell(hz(96 + (j % 3) * 4), 0.6), 0.10)
    else:
        seq = ["G5", "E5", "C5"]
        for i, n in enumerate(seq):
            _add(buf, 0.15 + i * 0.22, marimba(hz(midi(n)), 0.55), 0.5)
        for n in ("C5", "E5", "G5"):
            _add(buf, 0.85, bell(hz(midi(n)), 1.5), 0.22)
        for j in range(4):
            _add(buf, 1.2 + j * 0.3, bell(hz(103 - (j % 2) * 3), 0.6), 0.09)
    buf /= max(1e-6, np.abs(buf).max()) / 0.85
    write_wav(path, buf)