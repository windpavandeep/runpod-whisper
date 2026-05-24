import asyncio
import os
import tempfile
import traceback
from typing import Any

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading Whisper model {MODEL_NAME} on {DEVICE}…")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")


def _audio_suffix(audio_bytes: bytes) -> str:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF":
        return ".wav"
    if len(audio_bytes) >= 4 and audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".wav"


def transcribe_full_audio(audio_bytes: bytes) -> dict[str, Any]:
    if len(audio_bytes) < 1600:
        return {"transcript": "", "segments": []}

    temp_path = None
    try:
        suffix = _audio_suffix(audio_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        segments_iter, _info = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        segments: list[dict[str, Any]] = []
        parts: list[str] = []
        for index, seg in enumerate(segments_iter):
            text = seg.text.strip()
            if not text:
                continue
            parts.append(text)
            segments.append(
                {
                    "id": f"seg-{index}",
                    "startMs": int(seg.start * 1000),
                    "endMs": int(seg.end * 1000),
                    "text": text,
                    "speakerIndex": 0,
                }
            )

        return {"transcript": " ".join(parts).strip(), "segments": segments}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/transcribe/full")
async def transcribe_full(file: UploadFile = File(...)) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcript": "", "segments": []}

    try:
        result = await asyncio.to_thread(transcribe_full_audio, audio_bytes)
        print(
            f"Full transcribe: {len(audio_bytes)} bytes -> "
            f"{len(result.get('segments', []))} segments"
        )
        return result
    except Exception:
        traceback.print_exc()
        return {"error": "Transcription failed", "transcript": "", "segments": []}


# Legacy realtime chunk socket (optional; not used by hybrid frontend)
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket) -> None:
    import struct

    await websocket.accept()
    print("Client connected (legacy chunk WS)")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "bytes" not in message:
                continue

            raw = message["bytes"]
            if len(raw) < 5:
                continue

            chunk_id = struct.unpack(">I", raw[:4])[0]
            audio = raw[4:]
            print(f"chunk {chunk_id}: {len(audio)} bytes")

            try:
                audio_np = (
                    __import__("numpy")
                    .frombuffer(audio, dtype="int16")
                    .astype(__import__("numpy").float32)
                    / 32768.0
                )
                segments, _ = model.transcribe(
                    audio_np, beam_size=1, language="en", vad_filter=True
                )
                text = " ".join(s.text for s in segments).strip()
                if text:
                    await safe_send_json(
                        websocket, {"chunkId": chunk_id, "text": text}
                    )
            except Exception:
                traceback.print_exc()
    except WebSocketDisconnect:
        pass
    finally:
        print("connection closed")
