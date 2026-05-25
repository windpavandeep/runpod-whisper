import asyncio
import json
import os
import struct
import subprocess
import tempfile
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

SERVER_VERSION = "2026-03-24-webm-merge"

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16_000

# Must match frontend MEDIA_RECORDER_CHUNK_MS / WHISPER_WS_CHUNK_SEC
CHUNK_DURATION_MS = 2_000

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


def _webm_to_wav(webm_path: str, wav_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            webm_path,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            wav_path,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _transcribe_merged_webm(
    webm_path: str,
    wav_path: str,
    *,
    chunk_index: int,
    start_ms: int,
) -> dict[str, Any]:
    if not os.path.exists(webm_path) or os.path.getsize(webm_path) < 200:
        return {"transcript": "", "segments": []}

    try:
        _webm_to_wav(webm_path, wav_path)
        return _transcribe_wav_file(
            wav_path,
            time_offset_ms=start_ms,
            chunk_index=chunk_index,
        )
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg failed: {err}") from exc


def _transcribe_uploaded_bytes(
    audio_bytes: bytes,
    *,
    time_offset_ms: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    if len(audio_bytes) < 1600:
        return {"transcript": "", "segments": []}

    temp_path = None
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
        "protocol": "MediaRecorder webm fragments appended → ffmpeg → whisper",
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
    webm_path = os.path.join(tempfile.gettempdir(), f"printx_{session_id}.webm")
    wav_path = os.path.join(tempfile.gettempdir(), f"printx_{session_id}.wav")
    chunk_id = 0

    print(f"WS connected ({SERVER_VERSION}) session={session_id}")

    for path in (webm_path, wav_path):
        if os.path.exists(path):
            os.remove(path)

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
                    chunk_id += 1
                    start_ms = (chunk_id - 1) * CHUNK_DURATION_MS
                    try:
                        result = await asyncio.to_thread(
                            _transcribe_merged_webm,
                            webm_path,
                            wav_path,
                            chunk_index=chunk_id - 1,
                            start_ms=start_ms,
                        )
                        await safe_send_json(
                            websocket,
                            {
                                "chunkId": chunk_id,
                                "transcript": result.get("transcript", ""),
                                "segments": result.get("segments", []),
                            },
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        await safe_send_json(
                            websocket,
                            {"chunkId": chunk_id, "error": str(exc), "transcript": "", "segments": []},
                        )
                continue

            if "bytes" not in message:
                continue

            data = message["bytes"]
            if len(data) < 1:
                continue

            with open(webm_path, "ab") as f:
                f.write(data)

            chunk_id += 1
            chunk_index = chunk_id - 1
            start_ms = chunk_index * CHUNK_DURATION_MS
            print(f"chunk {chunk_id}: appended {len(data)} bytes → {webm_path}")

            try:
                result = await asyncio.to_thread(
                    _transcribe_merged_webm,
                    webm_path,
                    wav_path,
                    chunk_index=chunk_index,
                    start_ms=start_ms,
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
        for path in (webm_path, wav_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        print(f"WS closed session={session_id}")
