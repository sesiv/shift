from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from typing import Annotated
import tempfile
import os
import asyncio
from pywhispercpp.model import Model


app = FastAPI()

# Модельки брать тут: https://huggingface.co/ggerganov/whisper.cpp/tree/main
model = Model('hf_models/ggml-base-q8_0.bin', n_threads=4) # лучше использовать medium или large-v3-turbo

@app.post("/transcribe")
async def transcribe(audio_file: Annotated[UploadFile, File()]):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp:
            tmp_path = tmp.name
            audio_bytes = await audio_file.read()
            tmp.write(audio_bytes)

        # Запускаем блокирующую функцию транскрипции в отдельном потоке
        def do_transcribe():
            segments = model.transcribe(tmp_path, language="ru")
            return " ".join([segment.text for segment in segments])

        transcribed_text = await asyncio.to_thread(do_transcribe)

        return {'text': transcribed_text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)