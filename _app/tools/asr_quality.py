"""Lightweight acoustic quality checks for ASR word timestamps.

This module is a screening tool, not a phoneme recognizer. It uses ASR word
boundaries and the spelling of a word to estimate whether each word and vowel
has enough voiced, non-clipped signal to be worth listening to.
"""
import math
import re
import wave

import numpy as np

VOWELS = "аеёиоуыэюя"
_NORM_RE = re.compile(r"[^а-яё]", re.I)


def norm(text: str) -> str:
    return _NORM_RE.sub("", str(text).lower().replace("ё", "е"))


def load_audio(path):
    """Return mono float32 audio and sample rate from a PCM WAV."""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width: {sw}")
    a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:
        a = (a - 128.0) / 128.0
    else:
        a /= float(2 ** (8 * sw - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def _window(audio, sr, start, end):
    i0 = max(0, int(float(start) * sr))
    i1 = min(len(audio), max(i0 + 1, int(float(end) * sr)))
    return audio[i0:i1]


def _metrics(x, sr=16000):
    if len(x) == 0:
        return {"duration_s": 0.0, "rms_db": -120.0, "peak_db": -120.0, "clipped_ratio": 0.0,
                "voiced_ratio": 0.0}
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    # A frame is considered voiced enough for this screen when its RMS is at
    # least 8% of the word's own peak. This is intentionally conservative.
    hop = max(1, int(len(x) / 20))
    frames = [x[i:i + hop] for i in range(0, len(x), hop)]
    frms = np.asarray([np.sqrt(np.mean(f * f)) for f in frames if len(f)], dtype=float)
    voiced_ratio = float(np.mean(frms >= max(peak * 0.08, 1e-4))) if len(frms) else 0.0
    return {
        "duration_s": round(len(x) / float(sr), 3),
        "rms_db": round(20.0 * math.log10(max(rms, 1e-6)), 1),
        "peak_db": round(20.0 * math.log10(max(peak, 1e-6)), 1),
        "clipped_ratio": round(float(np.mean(np.abs(x) >= 0.98)), 4),
        "voiced_ratio": round(voiced_ratio, 3),
    }


def _flags(m, expected_duration=None):
    flags = []
    if m["rms_db"] < -38.0:
        flags.append("quiet")
    if m["clipped_ratio"] > 0.01:
        flags.append("clipped")
    if m["voiced_ratio"] < 0.25:
        flags.append("mostly_silent")
    if expected_duration is not None and m["duration_s"] > max(2.5, expected_duration * 2.2):
        flags.append("stretched")
    return flags


def vowel_quality(audio, sr, word, start, end):
    """Return measurements for each written vowel in an ASR word window.

    The vowel intervals are proportional character intervals. It is useful for
    finding quiet or clipped vowels and for making short listening clips, but
    it cannot establish stress or phoneme identity on its own.
    """
    text = norm(word)
    vowels = [(i, ch) for i, ch in enumerate(text) if ch in VOWELS]
    if not vowels or end <= start:
        return []
    span = float(end) - float(start)
    out = []
    for n, (i, ch) in enumerate(vowels):
        left = float(start) + span * (i / max(len(text), 1))
        right_index = vowels[n + 1][0] if n + 1 < len(vowels) else min(i + 1, len(text))
        right = float(start) + span * (right_index / max(len(text), 1))
        m = _metrics(_window(audio, sr, left, max(left + 0.025, right)), sr)
        m.update({"vowel": ch, "index": n, "start_s": round(left, 3), "end_s": round(right, 3)})
        m["flags"] = _flags(m)
        out.append(m)
    return out


def word_quality(audio, sr, item, expected=None):
    """Measure one ASR word and all of its written vowels."""
    start, end = float(item.get("s", 0.0)), float(item.get("e", 0.0))
    m = _metrics(_window(audio, sr, start, end), sr)
    expected_duration = None
    if expected:
        # A rough speech-rate prior; it only drives the stretched flag.
        expected_duration = max(0.08, len(norm(expected)) * 0.055)
    m["flags"] = _flags(m, expected_duration)
    m.update({"word": item.get("w", "").strip(), "start_s": round(start, 3), "end_s": round(end, 3)})
    m["vowels"] = vowel_quality(audio, sr, item.get("w", ""), start, end)
    return m


def summarize(rows):
    """Compact counts for a JSON report."""
    words = len(rows)
    bad_words = [r for r in rows if r.get("flags")]
    vowels = [v for r in rows for v in r.get("vowels", [])]
    bad_vowels = [v for v in vowels if v.get("flags")]
    return {
        "words": words,
        "words_with_flags": len(bad_words),
        "vowels": len(vowels),
        "vowels_with_flags": len(bad_vowels),
        "flagged_words": [{"word": r["word"], "t": r["start_s"], "flags": r["flags"]} for r in bad_words[:100]],
        "flagged_vowels": [{"word": r["word"], "vowel": v["vowel"], "t": v["start_s"], "flags": v["flags"]}
                           for r in rows for v in r.get("vowels", []) if v.get("flags")][:200],
    }
