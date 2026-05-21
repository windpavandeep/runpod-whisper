import numpy as np

from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Whisper model...")

model = WhisperModel(
    "tiny.en",
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

            audio_buffer += chunk

            # ~2 seconds of 16 kHz mono int16 PCM
            if len(audio_buffer) >= 16000 * 2 * 2:

                audio_np = np.frombuffer(
                    audio_buffer,
                    dtype=np.int16,
                ).astype(np.float32) / 32768.0

                segments, _ = model.transcribe(
                    audio_np,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                )

                text = ""

                for segment in segments:
                    text += segment.text + " "

                if text.strip():
                    await websocket.send_json({
                        "text": text.strip(),
                    })

                audio_buffer = b""

    except Exception as e:

        print(e)

    finally:

        print("client disconnected")
