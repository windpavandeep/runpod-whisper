import asyncio
import os
import struct
import tempfile
import traceback
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

SERVER_VERSION = "2026-03-24-wav-pcm"

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16_000

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

# Must match frontend WHISPER_WS_CHUNK_SEC
CHUNK_DURATION_MS = 5_000

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
    """Wrap raw int16 LE mono PCM in a valid WAV container for ffmpeg."""
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


def _transcribe_bytes_as_wav(
    audio_bytes: bytes,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    """Always write a .wav temp file — never .webm (av fails on mis-detected PCM)."""
    if len(audio_bytes) < 1600:
        return {"transcript": "", "segments": []}

    if _is_wav(audio_bytes):
        file_bytes = audio_bytes
    else:
        # WebSocket PCM, mis-tagged uploads, etc.
        file_bytes = _pcm_int16_to_wav_bytes(audio_bytes)

    temp_path = None
    try:
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


def transcribe_pcm_int16(
    pcm_bytes: bytes,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    return _transcribe_bytes_as_wav(
        pcm_bytes,
        time_offset_ms=time_offset_ms,
        chunk_index=chunk_index,
    )


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    return _transcribe_bytes_as_wav(
        audio_bytes,
        time_offset_ms=time_offset_ms,
        chunk_index=chunk_index,
    )


def transcribe_full_audio(audio_bytes: bytes) -> dict[str, Any]:
    return transcribe_audio_bytes(audio_bytes, time_offset_ms=0, chunk_index=0)


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
            transcribe_audio_bytes,
            audio_bytes,
            time_offset_ms=max(0, start_ms),
            chunk_index=max(0, chunk_index),
        )
        print(
            f"Chunk {chunk_index} @ {start_ms}ms: {len(audio_bytes)} bytes -> "
            f"{len(result.get('segments', []))} segments"
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
        result = await asyncio.to_thread(transcribe_full_audio, audio_bytes)
        print(
            f"Full transcribe: {len(audio_bytes)} bytes -> "
            f"{len(result.get('segments', []))} segments"
        )
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"error": str(exc), "transcript": "", "segments": []}


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket) -> None:
    import struct

    await websocket.accept()
    print(f"Client connected (WS {SERVER_VERSION})")

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
            if len(audio) < 1600:
                continue

            chunk_index = max(0, chunk_id - 1)
            start_ms = chunk_index * CHUNK_DURATION_MS
            print(
                f"chunk {chunk_id} @ {start_ms}ms: {len(audio)} bytes "
                f"(pcm, wrap wav)"
            )

            try:
                result = await asyncio.to_thread(
                    transcribe_pcm_int16,
                    audio,
                    time_offset_ms=start_ms,
                    chunk_index=chunk_index,
                )
                payload = {
                    "chunkId": chunk_id,
                    "transcript": result.get("transcript", ""),
                    "segments": result.get("segments", []),
                }
                if not await safe_send_json(websocket, payload):
                    break
            except Exception as exc:
                traceback.print_exc()
                await safe_send_json(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "error": str(exc),
                        "transcript": "",
                        "segments": [],
                    },
                )
    except WebSocketDisconnect:
        pass
    finally:
        print("connection closed")
