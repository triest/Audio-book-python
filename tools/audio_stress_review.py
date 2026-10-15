r"""Listening workflow for uncertain stress. Automatic stress verification from audio is NOT reliable (ASR
normalises words; alignment only proves a word was spoken), so this tool makes the human check quick.

  # 1) make A/B clips: one sentence per candidate reading, same voice, plus an index page
  .venv\Scripts\python.exe audio_stress_review.py make review_out --item "Он сказал, что замок пуст.|замок|з+амок,зам+ок"
  .venv\Scripts\python.exe audio_stress_review.py make review_out --from-json items.json   # [{"sentence","word","candidates":[..]}]
  # 2) open review_out\index.html, listen, then record the verdict (writes forced_stress_words.json + provenance log)
  .venv\Scripts\python.exe audio_stress_review.py accept замок зам+ок --context "Дверной замок заклинило."

Records go to stress_review_log.jsonl (word, chosen form, context, date, silero-stress version) - append-only, so a
decision can be traced and reverted; the dictionary file itself is rewritten atomically via a .tmp + os.replace."""
import argparse
import datetime
import html
import importlib.metadata as md
import json
import os
import re
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
LOG = HERE / "stress_review_log.jsonl"
SR = 48000


def mark_word(sentence: str, word: str, form: str) -> str:
    pat = re.compile(r"(?<![А-Яа-яЁё])" + re.escape(word) + r"(?![А-Яа-яЁё])", re.I)
    return pat.sub(lambda m: form[0].upper() + form[1:] if m.group(0)[:1].isupper() else form, sentence, count=1)


def make(out: Path, items: list, speaker: str = "aidar"):
    import numpy as np
    import torch
    imp = torch.package.PackageImporter(str(HERE / "silero_model_v5_5_ru.pt"))
    model = imp.load_pickle("tts_models", "model")
    model.to("cpu")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, it in enumerate(items, 1):
        row = {"sentence": it["sentence"], "word": it["word"], "clips": []}
        for k, form in enumerate(it["candidates"], 1):
            text = mark_word(it["sentence"], it["word"], form)
            audio = model.apply_tts(text=text, speaker=speaker, sample_rate=SR, put_accent=False, put_yo=False).numpy()
            fn = f"{n:03d}_{k}_{re.sub(r'[^0-9A-Za-zА-Яа-я]', '_', form)}.wav"
            with wave.open(str(out / fn), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            row["clips"].append({"form": form, "text": text, "file": fn})
        rows.append(row)
    body = "".join(
        f"<h3>{n}. {html.escape(r['word'])}</h3><p>{html.escape(r['sentence'])}</p>" + "".join(
            f"<p><b>{html.escape(c['form'])}</b> <audio controls src='{c['file']}'></audio> <code>{html.escape(c['text'])}</code></p>"
            for c in r["clips"]) +
        f"<p><i>accept: <code>audio_stress_review.py accept {html.escape(r['word'])} &lt;form&gt; --context \"{html.escape(r['sentence'])}\"</code></i></p>"
        for n, r in enumerate(rows, 1))
    (out / "index.html").write_text("<meta charset=utf-8><title>Stress review</title>" + body, encoding="utf-8")
    (out / "items.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} items -> {out / 'index.html'}")


def accept(word: str, form: str, context: str = ""):
    import fb2_reader as fr
    path = Path(fr.DEFAULT_FORCED_STRESS_PATH)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if form.replace("+", "").lower() != word.lower() and form.replace("+", "").replace("ё", "е") != word.lower().replace("ё", "е"):
        sys.exit(f"form {form!r} does not spell {word!r}")
    previous = data.get(word.lower())
    data[word.lower()] = form
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    try:
        ver = md.version("silero-stress")
    except Exception:
        ver = None
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.datetime.now().isoformat(timespec="seconds"), "word": word.lower(),
                            "form": form, "previous": previous, "context": context, "silero_stress": ver,
                            "source": "listening review"}, ensure_ascii=False) + "\n")
    print(f"{word} -> {form} saved to {path} (logged in {LOG.name}; previous: {previous})")
    if context:
        print("NOTE: a dictionary entry applies to ALL occurrences; for a homograph use a phrase key instead.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("out"); m.add_argument("--item", action="append", default=[])
    m.add_argument("--from-json"); m.add_argument("--speaker", default="aidar")
    a = sub.add_parser("accept"); a.add_argument("word"); a.add_argument("form"); a.add_argument("--context", default="")
    args = ap.parse_args()
    if args.cmd == "make":
        items = []
        for s in args.item:
            sentence, word, cands = s.split("|")
            items.append({"sentence": sentence, "word": word, "candidates": cands.split(",")})
        if args.from_json:
            items += json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        make(Path(args.out), items, args.speaker)
    else:
        accept(args.word, args.form, args.context)


if __name__ == "__main__":
    main()
