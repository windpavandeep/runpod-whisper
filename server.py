import asyncio
import struct
import traceback

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Whisper model...")

model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="float16",
)

print("Whisper model loaded!")


def transcribe_chunk(audio_bytes: bytes) -> str:
    if len(audio_bytes) < 1600:
        return ""

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


def parse_chunk_message(data: bytes) -> tuple[int, bytes]:
    if len(data) < 5:
        raise ValueError("Chunk too short")

    chunk_id = struct.unpack(">I", data[:4])[0]
    audio = data[4:]
    return chunk_id, audio


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    """Return False if the client already closed the socket."""
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("Client connected")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" not in message:
                continue

            raw = message["bytes"]

            try:
                chunk_id, audio = parse_chunk_message(raw)
            except ValueError as exc:
                print("Invalid chunk:", exc)
                continue

            print(f"chunk {chunk_id}: {len(audio)} bytes")

            try:
                text = await asyncio.to_thread(transcribe_chunk, audio)
            except Exception:
                print(f"Transcribe error (chunk {chunk_id}):")
                traceback.print_exc()
                continue

            if not text:
                continue

            print(f"chunk {chunk_id} transcript:", text)
            if not await safe_send_json(
                websocket,
                {"chunkId": chunk_id, "text": text},
            ):
                print(f"Client gone before chunk {chunk_id} reply was sent")
                break

    except WebSocketDisconnect as exc:
        print(f"Client disconnected (code={exc.code})")

    except Exception:
        print("WebSocket handler error:")
        traceback.print_exc()

    finally:
        print("connection closed")
