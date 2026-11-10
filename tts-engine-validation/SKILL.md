---
name: tts-engine-validation
description: Validate a new Russian TTS engine in the audiobook project, local or cloud API. Use when adding, repairing, or comparing a REST TTS service, especially when GPU usage, long chapter generation, Russian stress accuracy, or provider credentials are in question.
---

# TTS engine validation

Use this workflow before accepting a Russian TTS engine into the GUI. Separate code-path verification from live provider synthesis: a mocked HTTP response or successful health check does not prove that a real account can synthesize audio.

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

## Cloud API

- Verify the provider's current API endpoint, request schema, authentication, response format, and text-size limit from current documentation.
- Confirm account signup and billing are available for the user's region before integrating. Record free quotas, currency, and payment methods as provider claims until verified in an account.
- Prefer standard-library HTTP calls when the API does not require a new package. Add any required package only through `install.bat`.
- Keep keys out of logs and source control. Read credentials from local GUI settings or an environment variable.
- Split text under the documented limit with a safety margin. Ensure the splitter also bounds unusually long sentences and single tokens; a sentence-only splitter may exceed the provider cap.
- Check audio content type/signature, reject JSON error bodies, and verify chapter output and metadata. Test multi-chunk audio joining.
- If no API key/voice ID is available, run mocked request/response and chunking checks, but report live synthesis, provider quality, billing, and stress behavior as unverified.

## GenVoice finding

GenVoice documents synchronous `POST https://api.genvoice.ru/v1/api/tts`, Bearer authentication, MP3 output, and a 500 source-character cap. Its stress guidance uses `+` before the stressed vowel. The project integration sends the normal stress pipeline's marked text, caps requests at 480 characters, and joins returned MP3 chunks. The REST docs do not specify pause-marker syntax, so do not invent or send editor-only pause markup. Without a valid API key and voice ID, the integration is code-path verified only; do not describe audio quality, actual billing, or live provider access as tested.

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
