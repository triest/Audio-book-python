---
name: tts-engine-validation
description: Validate a new local Russian TTS engine in the audiobook project. Use when adding, repairing, or comparing a REST TTS service, especially when GPU usage, long chapter generation, or Russian stress accuracy is in question.
---

# TTS engine validation

Use this workflow before accepting a local TTS engine into the GUI.

## Installation

- Put every dependency and model download in `install.bat`.
- Use a separate Windows venv when torch or package versions can conflict.
- Verify `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()`.
- Do not silently accept a CPU fallback on a CUDA machine.

## REST service

- Start the service in its intended venv.
- Check `/health` for `model_loaded`, engine name, service version, voices, and device.
- Send one short Russian request and verify HTTP 200, a nonempty WAV, and the expected sample rate.
- Read the latest service log entries by timestamp. A listening port alone does not prove inference works.
- After changing service code, bump `SERVICE_VERSION` in the service and the matching GUI expected version, then kill the old process on the port.

## Audiobook check

- Run the preface and chapter 1 from a real FB2 into a separate output directory.
- Check that chapter files are nonempty and failures do not become silent placeholder audio.
- Probe homographs such as `глаза`, `замок`, `мука`, `атлас`, and `звонит`.
- Compare the text sent to the engine with the stress-marked text produced by the project pipeline.

## Decision

- **Accept**: GPU works, the short request works, chapter 1 completes, and Russian stress is usable.
- **Hold**: the engine runs but needs a known setup fix or a manual voice profile.
- **Reject for audiobook use**: inference works but the engine ignores stress control or repeatedly misreads Russian homographs. Keep it only as an experimental mode if explicitly requested.

## F5-TTS-Russian finding

The `hotstone228/F5-TTS-Russian` checkpoint can run on CUDA, but it receives ordinary text and does not honor the project's `+` stress markers. A successful WAV response does not prove correct Russian pronunciation. Do not use it as the primary Russian audiobook engine when homograph stress matters. Keep ESpeech RL-V2 as the stress-aware reference path.

## Reporting

Record the engine, venv, port, torch/CUDA result, short-request result, chapter result, stress result, and final decision in `CLAUDE.md` under the considered or deferred engines section.
