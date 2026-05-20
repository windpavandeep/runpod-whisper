from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

model = WhisperModel(
    "distil-large-v3",
    device="cuda",
    compute_type="float16"
)

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        audio_bytes = await websocket.receive_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        segments, info = model.transcribe(
            temp_path,
            beam_size=1
        )

        text = ""

        for segment in segments:
            text += segment.text

        os.remove(temp_path)

        await websocket.send_json({
            "text": text
        })