from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8"
)

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("Client connected")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            segments, info = model.transcribe(
                temp_path,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = ""

            for segment in segments:
                text += segment.text

            os.remove(temp_path)

            await websocket.send_json({
                "text": text.strip()
            })

    except Exception as e:
        print("Error:", e)

    finally:
        print("Client disconnected")