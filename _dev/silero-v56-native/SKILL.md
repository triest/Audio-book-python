---
name: silero-v56-native
description: Run and troubleshoot Russian audiobook synthesis through Silero v5_6_ru using the model's native stress and homograph handling, retaining only explicit manual word overrides. Use when starting a book or chapter in the GUI or CLI, comparing Silero quality with Balabolka, or investigating stretching, swallowed words, pauses, and wrong stress.
---

# Silero v5_6 native workflow

## Default policy

- For `v5_6_ru`, use `--stress-engine none`; the GUI applies the same rule automatically.
- Keep explicit entries from `forced_stress_words.json`, `manual_stress_overrides.json`, and the book's `*.manual_stress_overrides.json`.
- Do not add phonetic substitutions for ordinary Russian words. Keep `_app/pronunciation_substitutions.json` for technical acronyms and documented foreign terms only.
- Preserve source spelling. The current TTS-only compound normalization is `во-первых` → `воп+ервых` and `во-вторых` → `вовтор+ых`.
- Keep ASR auto-repair enabled for normal runs. Disable it only for a diagnostic run when the ASR worker/GPU is failing.

## CLI run

Use a fresh, explicit output directory so a GUI run cannot be mistaken for a CLI run:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe _app\fb2_reader.py pepel-pobedy.fb2 `
  --mode silero_rest --rest-url http://localhost:5010 `
  --model v5_6_ru --speaker xenia --sample-rate 48000 `
  --sentence-break-ms 280 --paragraph-break-ms 490 --comma-break-ms 120 `
  --outdir audiobook_output_native --chapters 1,2 --stress-engine none
```

The preface is chapter index 1; `Глава 1` is index 2. Never reuse a directory currently selected by the GUI.

## Verification

- Check the output WAVs and `silero_rest_client.log`.
- Confirm metadata has `model: v5_6_ru` and `stress_engine: none`.
- ASR checks missing words and gaps; it cannot verify lexical stress. Verify stress by listening to an A/B sample.
- If a run stalls, inspect the process and service port 5010 before starting another run.

## Known failure mode

Client-side `silero-stress` plus `server_accent=false` can make good words sound stretched or overly soft. For `v5_6_ru`, return to the native path before changing text or adding substitutions.
