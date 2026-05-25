import asyncio
import json
import os
import struct
import tempfile
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

SERVER_VERSION = "2026-03-24-pcm-wav-ws"

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16_000

# Must match frontend WHISPER_WS_CHUNK_SEC
CHUNK_DURATION_MS = 3_000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading Whisper model {MODEL_NAME} on {DEVICE}… ({SERVER_VERSION})")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")

WHISPER_OPTS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": True,
}


def _is_wav(audio_bytes: bytes) -> bool:
    return (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )


def _pcm_int16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    num_samples = len(pcm_bytes) // 2
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _parse_pcm_ws_frame(data: bytes) -> tuple[int, bytes] | None:
    """Binary frame: [4-byte BE chunkId][int16 PCM little-endian]."""
    if len(data) < 8:
        return None
    chunk_id = int.from_bytes(data[:4], byteorder="big", signed=False)
    pcm = data[4:]
    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]
    if len(pcm) < 2:
        return None
    return chunk_id, pcm


def _segments_from_whisper(
    segments_iter: Any,
    *,
    time_offset_ms: int = 0,
    id_prefix: str = "seg",
    speaker_index: int = 0,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    parts: list[str] = []
    for index, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            continue
        parts.append(text)
        segments.append(
            {
                "id": f"{id_prefix}-{index}",
                "startMs": int(seg.start * 1000) + time_offset_ms,
                "endMs": int(seg.end * 1000) + time_offset_ms,
                "text": text,
                "speakerIndex": speaker_index,
            }
        )
    return {"transcript": " ".join(parts).strip(), "segments": segments}


def _transcribe_wav_file(
    wav_path: str,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    segments_iter, _info = model.transcribe(wav_path, **WHISPER_OPTS)
    return _segments_from_whisper(
        segments_iter,
        time_offset_ms=time_offset_ms,
        id_prefix=f"chunk-{chunk_index}",
        speaker_index=chunk_index % 2,
    )


def _transcribe_pcm_bytes(
    pcm_bytes: bytes,
    *,
    chunk_id: int,
) -> dict[str, Any]:
    min_bytes = SAMPLE_RATE * 2 // 2
    if len(pcm_bytes) < min_bytes:
        return {"transcript": "", "segments": []}

    chunk_index = max(0, chunk_id - 1)
    start_ms = chunk_index * CHUNK_DURATION_MS
    wav_bytes = _pcm_int16_to_wav_bytes(pcm_bytes)

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            temp_path = tmp.name

        return _transcribe_wav_file(
            temp_path,
            time_offset_ms=start_ms,
            chunk_index=chunk_index,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _transcribe_uploaded_bytes(
    audio_bytes: bytes,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    if len(audio_bytes) < 1600:
        return {"transcript": "", "segments": []}

    temp_path: str | None = None
    try:
        if _is_wav(audio_bytes):
            file_bytes = audio_bytes
        else:
            file_bytes = _pcm_int16_to_wav_bytes(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        return _transcribe_wav_file(
            temp_path,
            time_offset_ms=time_offset_ms,
            chunk_index=chunk_index,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "printx-whisper",
        "version": SERVER_VERSION,
        "health": "/health",
        "websocket": "/ws/transcribe",
        "protocol": "binary [uint32 BE chunkId][int16 PCM] → WAV per chunk → whisper",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "version": SERVER_VERSION}


@app.post("/transcribe/chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(0),
    start_ms: int = Form(0),
) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcript": "", "segments": [], "chunkIndex": chunk_index}

    try:
        result = await asyncio.to_thread(
            _transcribe_uploaded_bytes,
            audio_bytes,
            time_offset_ms=max(0, start_ms),
            chunk_index=max(0, chunk_index),
        )
        result["chunkIndex"] = chunk_index
        return result
    except Exception as exc:
        traceback.print_exc()
        return {
            "error": str(exc),
            "transcript": "",
            "segments": [],
            "chunkIndex": chunk_index,
        }


@app.post("/transcribe/full")
async def transcribe_full(file: UploadFile = File(...)) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcript": "", "segments": []}

    try:
        return await asyncio.to_thread(_transcribe_uploaded_bytes, audio_bytes)
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc), "transcript": "", "segments": []}


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = uuid.uuid4().hex[:12]
    print(f"WS connected ({SERVER_VERSION}) session={session_id}")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                try:
                    meta = json.loads(message["text"])
                except json.JSONDecodeError:
                    meta = {}
                if meta.get("type") == "end":
                    continue

            if "bytes" not in message:
                continue

            data = message["bytes"]
            parsed = _parse_pcm_ws_frame(data)
            if parsed is None:
                await safe_send_json(
                    websocket,
                    {
                        "error": (
                            "Invalid PCM frame. Expected [4-byte chunkId][int16 PCM]. "
                            "Do not send WebM fragments — redeploy this server and refresh the app."
                        ),
                        "transcript": "",
                        "segments": [],
                    },
                )
                continue

            chunk_id, pcm_bytes = parsed
            print(f"chunk {chunk_id}: {len(pcm_bytes)} pcm bytes")

            try:
                result = await asyncio.to_thread(
                    _transcribe_pcm_bytes,
                    pcm_bytes,
                    chunk_id=chunk_id,
                )
                if not await safe_send_json(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "transcript": result.get("transcript", ""),
                        "segments": result.get("segments", []),
                    },
                ):
                    break
            except Exception as exc:
                traceback.print_exc()
                if not await safe_send_json(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "error": str(exc),
                        "transcript": "",
                        "segments": [],
                    },
                ):
                    break

    except WebSocketDisconnect:
        pass
    finally:
        print(f"WS closed session={session_id}")
