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

            if not audio_bytes:
                continue

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                segments, _info = model.transcribe(
                    temp_path,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )

                text = "".join(segment.text for segment in segments).strip()

                await websocket.send_json({"text": text})

            except Exception as e:
                print("Transcription error:", e)
                await websocket.send_json({"text": "", "error": str(e)})

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    except Exception as e:
        print("Connection error:", e)

    finally:
        print("Client disconnected")
