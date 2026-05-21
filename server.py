import os
import uuid
import asyncio
import numpy as np
import soundfile as sf

from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Whisper model...")

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

print("Model loaded!")

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):

    await websocket.accept()

    print("Client connected")

    audio_chunks = []

    try:
        while True:

            data = await websocket.receive_bytes()

            if data == b"END":

                audio_np = np.frombuffer(
                    b"".join(audio_chunks),
                    dtype=np.int16
                )

                wav_path = f"{TEMP_DIR}/{uuid.uuid4()}.wav"

                sf.write(
                    wav_path,
                    audio_np,
                    16000
                )

                segments, info = model.transcribe(
                    wav_path,
                    beam_size=5
                )

                final_text = ""

                for segment in segments:
                    final_text += segment.text + " "

                await websocket.send_json({
                    "text": final_text.strip(),
                    "language": info.language
                })

                os.remove(wav_path)

                audio_chunks = []

            else:
                audio_chunks.append(data)

    except Exception as e:
        print(e)

    finally:
        print("Client disconnected")
