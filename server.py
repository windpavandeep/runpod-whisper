import asyncio
import json
import os
import struct
import tempfile
import traceback
import uuid
import subprocess
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

SERVER_VERSION = "2026-05-26-realtime-fixed"

MODEL_NAME = "large-v3"

# CUDA fallback support
try:
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("CUDA model loaded")
except Exception:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    print("CPU fallback model loaded")

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 3000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_OPTS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
}


def _is_wav(audio_bytes: bytes) -> bool:
    return (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )


def convert_to_wav(input_path: str) -> str:
    output_path = f"{input_path}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return output_path


def pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]

    data_size = len(pcm_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SAMPLE_RATE,
        SAMPLE_RATE * 2,
        2,
        16,
        b"data",
        data_size,
    )

    return header + pcm_bytes


def _looks_like_webm(data: bytes) -> bool:
    """EBML header — indicates MediaRecorder/webm was sent instead of PCM."""
    return len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3"


def parse_pcm_frame(data: bytes):
    if len(data) < 8:
        return None

    if _looks_like_webm(data[4:]) or _looks_like_webm(data):
        return None

    chunk_id = int.from_bytes(data[:4], "big")
    pcm = data[4:]

    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]

    return chunk_id, pcm


def transcribe_file(
    path: str,
    *,
    chunk_index: int = 0,
    time_offset_ms: int = 0,
):
    segments_iter, info = model.transcribe(
        path,
        **WHISPER_OPTS,
    )

    segments = []
    transcript_parts = []

    for index, seg in enumerate(segments_iter):
        text = seg.text.strip()

        if not text:
            continue

        transcript_parts.append(text)

        segments.append(
            {
                "id": f"chunk-{chunk_index}-{index}",
                "startMs": int(seg.start * 1000) + time_offset_ms,
                "endMs": int(seg.end * 1000) + time_offset_ms,
                "text": text,
                "speakerIndex": 0,
            }
        )

    return {
        "transcript": " ".join(transcript_parts),
        "segments": segments,
    }


async def safe_send_json(websocket: WebSocket, payload: dict):
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


@app.get("/")
async def root():
    return {
        "service": "whisper-realtime",
        "version": SERVER_VERSION,
        "device": DEVICE,
        "websocket": "/ws/transcribe",
        "protocol": "binary [uint32 BE chunkId][int16 PCM 16kHz mono] → in-memory WAV → whisper",
        "chunk_duration_ms": CHUNK_DURATION_MS,
        "note": "WebSocket path does not use ffmpeg; do not send MediaRecorder webm fragments.",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "version": SERVER_VERSION,
        "protocol": "pcm-ws",
        "chunk_duration_ms": CHUNK_DURATION_MS,
    }


@app.post("/transcribe/full")
async def transcribe_full(file: UploadFile = File(...)):
    temp_input = None
    temp_wav = None

    try:
        audio_bytes = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file.filename,
        ) as tmp:
            tmp.write(audio_bytes)
            temp_input = tmp.name

        temp_wav = convert_to_wav(temp_input)

        result = await asyncio.to_thread(
            transcribe_file,
            temp_wav,
        )

        return result

    except Exception as exc:
        traceback.print_exc()

        return {
            "error": str(exc),
            "transcript": "",
            "segments": [],
        }

    finally:
        for path in [temp_input, temp_wav]:
            if path and os.path.exists(path):
                os.remove(path)


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = uuid.uuid4().hex[:8]

    print(f"WS connected {session_id}")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" not in message:
                continue

            data = message["bytes"]

            parsed = parse_pcm_frame(data)

            if parsed is None:
                hint = (
                    "Invalid PCM frame. Expected [4-byte chunkId][int16 PCM]. "
                    "Do not send MediaRecorder webm blobs."
                )
                if _looks_like_webm(data):
                    hint = (
                        "Received WebM/EBML data, not PCM. "
                        "Deploy the AudioWorklet frontend and server "
                        f"{SERVER_VERSION}."
                    )
                await safe_send_json(
                    websocket,
                    {
                        "chunkId": 0,
                        "error": hint,
                        "transcript": "",
                        "segments": [],
                    },
                )
                continue

            chunk_id, pcm_bytes = parsed

            if len(pcm_bytes) < 3200:
                continue

            chunk_index = max(0, chunk_id - 1)
            start_ms = chunk_index * CHUNK_DURATION_MS

            wav_bytes = pcm_to_wav_bytes(pcm_bytes)

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                ) as tmp:
                    tmp.write(wav_bytes)
                    temp_path = tmp.name

                result = await asyncio.to_thread(
                    transcribe_file,
                    temp_path,
                    chunk_index=chunk_index,
                    time_offset_ms=start_ms,
                )

                ok = await safe_send_json(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "transcript": result["transcript"],
                        "segments": result["segments"],
                    },
                )

                if not ok:
                    break

            except Exception as exc:
                traceback.print_exc()

                ok = await safe_send_json(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "error": str(exc),
                        "transcript": "",
                        "segments": [],
                    },
                )

                if not ok:
                    break

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    except WebSocketDisconnect:
        print(f"WS disconnected {session_id}")

    except Exception:
        traceback.print_exc()

    finally:
        print(f"WS closed {session_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
    
    
    
    