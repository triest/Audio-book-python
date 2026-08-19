from fastapi import FastAPI, HTTPException
from starlette.responses import Response
import uvicorn
import torch
from ruaccent import RUAccent
import os
import re
from num2words import num2words  # Библиотека для преобразования чисел в слова
from xml.sax.saxutils import escape as xml_escape

app = FastAPI()

version = "1.1"
model = None
accentizer = None


@app.on_event("startup")
async def startup_event():
    global model, accentizer
    modelurl = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'silero_model.pt'

    if not os.path.isfile(local_file):
        print("Downloading Silero TTS model...")
        torch.hub.download_url_to_file(modelurl, local_file)

    try:
        model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        model.to(device)
        print("TTS Model loaded successfully")
    except Exception as e:
        print(f"Failed to load TTS model: {e}")
        model = None

    try:
        accentizer = RUAccent()
        accentizer.load(omograph_model_size='turbo', use_dictionary=True)
        print("RUAccent model loaded successfully")
    except Exception as e:
        print(f"Failed to load RUAccent model: {e}")


def preprocess_text(text):
    """Преобразует цифры в текстовый формат."""
    words = text.split()
    processed_words = []
    for word in words:
        if word.isdigit():
            try:
                word = num2words(int(word), lang='ru')
            except Exception as e:
                print(f"Failed to convert number {word} to words: {e}")
        processed_words.append(word)
    return " ".join(processed_words)


@app.get(
    "/getwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response
)
async def getwav(text_to_speech: str, speaker: str = "xenia", sample_rate: int = 24000):
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded")

    preprocessed_text = preprocess_text(text_to_speech)
    accented_text = accentizer.process_all(preprocessed_text) if accentizer else preprocessed_text
    print(f"Text after accent processing: {accented_text}")

    wavfile = "temp.wav"
    path = model.save_wav(text=accented_text, speaker=speaker, sample_rate=sample_rate)

    with open(path, "rb") as in_file:
        data = in_file.read()

    return Response(content=data, media_type="audio/wav")


# --------------------------------------------------------------------------
# SSML: интонационные паузы, вопросы/восклицания, границы предложений/абзацев
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_SOFT_PAUSE_RE = re.compile(r"([,;:]|—|--)\s+")


def _accent_plain(text: str) -> str:
    """Прогоняет обычный (не-SSML) текст через preprocess_text + ударения."""
    preprocessed = preprocess_text(text)
    return accentizer.process_all(preprocessed) if accentizer else preprocessed


def build_ssml(text: str, sentence_break_ms: int = 320,
                paragraph_break_ms: int = 550, comma_break_ms: int = 180) -> str:
    """Строит SSML-документ для Silero из обычного текста.

    * Каждый абзац -> <p>, между абзацами длинная пауза.
    * Каждое предложение -> <s>, между предложениями пауза покороче;
      вопросительный/восклицательный знак сохраняется как есть, что
      Silero использует для интонации вопроса/восклицания.
    * Внутри предложения после запятых/тире/двоеточий/точек с запятой —
      небольшая пауза <break/>, имитирующая естественную интонационную
      паузу при чтении.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    p_chunks = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        s_chunks = []
        for sent in sentences:
            accented = _accent_plain(sent)
            escaped = xml_escape(accented)
            # небольшие паузы внутри предложения на знаках препинания
            escaped = _SOFT_PAUSE_RE.sub(
                lambda m: f'{xml_escape(m.group(1))}<break time="{comma_break_ms}ms"/> ',
                escaped,
            )
            s_chunks.append(f"<s>{escaped}</s>")
        if s_chunks:
            joiner = f'<break time="{sentence_break_ms}ms"/>'
            p_chunks.append("<p>" + joiner.join(s_chunks) + "</p>")

    joiner = f'<break time="{paragraph_break_ms}ms"/>'
    body = joiner.join(p_chunks)
    return f"<speak>{body}</speak>"


@app.get(
    "/getssmlwav",
    responses={200: {"content": {"audio/wav": {}}}},
    response_class=Response
)
async def getssmlwav(text_to_speech: str, speaker: str = "xenia", sample_rate: int = 24000,
                      sentence_break_ms: int = 320, paragraph_break_ms: int = 550,
                      comma_break_ms: int = 180, raw_ssml: bool = False):
    """Синтез речи с интонационными паузами и вопросительной/восклицательной
    интонацией.

    - text_to_speech: обычный текст (будет автоматически превращён в SSML
      с паузами на знаках препинания и границах предложений/абзацев), либо
      уже готовый SSML-документ, если raw_ssml=true.
    - raw_ssml: если true, text_to_speech передаётся модели как есть
      (ожидается валидный <speak>...</speak>).
    """
    if model is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded")

    if raw_ssml:
        ssml = text_to_speech
    else:
        ssml = build_ssml(
            text_to_speech,
            sentence_break_ms=sentence_break_ms,
            paragraph_break_ms=paragraph_break_ms,
            comma_break_ms=comma_break_ms,
        )

    print(f"SSML for synthesis: {ssml}")

    try:
        path = model.save_wav(ssml_text=ssml, speaker=speaker, sample_rate=sample_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SSML synthesis failed: {e}")

    with open(path, "rb") as in_file:
        data = in_file.read()

    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run("silero_rest_service:app", host="0.0.0.0", port=5010, log_level="info")
