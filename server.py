import numpy as np

from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16",
)

print("Whisper model loaded!")


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    audio_buffer = b""

    try:

        while True:

            chunk = await websocket.receive_bytes()

            print("received bytes:", len(chunk))

            audio_buffer += chunk

            if len(audio_buffer) > 32000 * 5:

                audio_np = np.frombuffer(
                    audio_buffer,
                    dtype=np.int16,
                ).astype(np.float32) / 32768.0

                segments, info = model.transcribe(
                    audio_np,
                    language="en",
                )

                text = ""

                for segment in segments:
                    text += segment.text + " "

                await websocket.send_json({
                    "text": text.strip(),
                })

                audio_buffer = b""

    except Exception as e:

        print(e)

    finally:

        print("client disconnected")
