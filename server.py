import asyncio
import traceback

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI()

WINDOW_BYTES = 16000 * 2 * 2  # ~2 sec of 16 kHz mono int16

print("Loading Whisper model...")

model = WhisperModel(
    "tiny.en",
    device="cuda",
    compute_type="float16",
)

print("Whisper model loaded!")


def transcribe_window(audio_bytes: bytes) -> str:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(
        np.float32
    ) / 32768.0

    segments, _ = model.transcribe(
        audio_np,
        beam_size=1,
        language="en",
        vad_filter=True,
    )

    return " ".join(segment.text for segment in segments).strip()


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Client connected")

    audio_buffer = b""

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" not in message:
                continue

            chunk = message["bytes"]
            print("received bytes:", len(chunk))
            audio_buffer += chunk

            if len(audio_buffer) < WINDOW_BYTES:
                continue

            window = audio_buffer[:WINDOW_BYTES]
            audio_buffer = audio_buffer[WINDOW_BYTES:]

            try:
                text = await asyncio.to_thread(transcribe_window, window)
                if text:
                    print("transcript:", text)
                    await websocket.send_json({"text": text})
            except Exception:
                print("Transcribe error:")
                traceback.print_exc()
                # Keep connection alive; skip bad window

    except WebSocketDisconnect as exc:
        print(f"Client disconnected (code={exc.code})")

    except Exception:
        print("WebSocket handler error:")
        traceback.print_exc()

    finally:
        print("connection closed")
