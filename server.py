import asyncio
import os
import struct
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

SERVER_VERSION = "2026-05-26-temp-audio-dir"

MODEL_NAME = "large-v3"

# Dedicated temp dir (override on RunPod: WHISPER_TEMP_AUDIO_DIR=/root/runpod-whisper/temp_audio)
_DEFAULT_TEMP_DIR = Path(__file__).resolve().parent / "temp_audio"
TEMP_AUDIO_DIR = Path(
    os.environ.get("WHISPER_TEMP_AUDIO_DIR", str(_DEFAULT_TEMP_DIR)),
).resolve()

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 3000

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


def ensure_temp_audio_dir() -> Path:
    """Create project temp folder (avoid /tmp permission/sandbox issues)."""
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(TEMP_AUDIO_DIR, 0o777)
    return TEMP_AUDIO_DIR


def _temp_wav_path() -> str:
    return str(TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}.wav")


def _temp_upload_path(suffix: str) -> str:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return str(TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}{safe_suffix}")


def _log_temp_file(path: str, *, chunk_id: int | None = None) -> int:
    size = os.path.getsize(path) if os.path.exists(path) else 0
    label = f"chunk {chunk_id}" if chunk_id is not None else "upload"
    print(f"TEMP PATH ({label}): {path}")
    print(f"FILE SIZE ({label}): {size}")
    if size < 50_000:
        print(
            f"WARNING ({label}): small WAV ({size} bytes) — "
            "expect ~96000+ for 3s PCM; frontend data may be broken.",
        )
    return size


def _remove_temp(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            print(f"TEMP cleanup failed {path}: {exc}")


def _is_wav(audio_bytes: bytes) -> bool:
    return (
        len(audio_bytes) >= 12
        and audio_bytes[:4] == b"RIFF"
        and audio_bytes[8:12] == b"WAVE"
    )


def convert_to_wav(input_path: str) -> str:
    base = os.path.basename(input_path)
    output_path = str(TEMP_AUDIO_DIR / f"{base}.wav")

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
    size = _log_temp_file(path, chunk_id=chunk_index)
    if size < 44:
        raise ValueError(f"WAV file too small to transcribe: {size} bytes")

    segments_iter, _info = model.transcribe(
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


@app.on_event("startup")
async def startup() -> None:
    path = ensure_temp_audio_dir()
    print(f"TEMP_AUDIO_DIR: {path} (writable: {os.access(path, os.W_OK)})")
    tmp_stat = os.stat("/tmp") if os.path.exists("/tmp") else None
    if tmp_stat:
        mode = oct(tmp_stat.st_mode)[-4:]
        sticky = bool(tmp_stat.st_mode & 0o1000)
        print(f"/tmp mode={mode} sticky={'yes' if sticky else 'NO (expected t in drwxrwxrwt)'}")


@app.get("/")
async def root():
    return {
        "service": "whisper-realtime",
        "version": SERVER_VERSION,
        "device": DEVICE,
        "temp_audio_dir": str(TEMP_AUDIO_DIR),
        "websocket": "/ws/transcribe",
        "protocol": "binary [uint32 BE chunkId][int16 PCM 16kHz mono] → WAV in temp_audio → whisper",
        "chunk_duration_ms": CHUNK_DURATION_MS,
    }


@app.get("/health")
async def health():
    ensure_temp_audio_dir()
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "version": SERVER_VERSION,
        "protocol": "pcm-ws",
        "chunk_duration_ms": CHUNK_DURATION_MS,
        "temp_audio_dir": str(TEMP_AUDIO_DIR),
        "temp_dir_writable": os.access(TEMP_AUDIO_DIR, os.W_OK),
    }


@app.post("/transcribe/full")
async def transcribe_full(file: UploadFile = File(...)):
    temp_input: str | None = None
    temp_wav: str | None = None

    try:
        audio_bytes = await file.read()
        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        temp_input = _temp_upload_path(suffix)

        with open(temp_input, "wb") as tmp:
            tmp.write(audio_bytes)

        _log_temp_file(temp_input)
        temp_wav = convert_to_wav(temp_input)
        _log_temp_file(temp_wav)

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
        _remove_temp(temp_input)
        _remove_temp(temp_wav)


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
            temp_path = _temp_wav_path()

            try:
                with open(temp_path, "wb") as tmp:
                    tmp.write(wav_bytes)

                _log_temp_file(temp_path, chunk_id=chunk_id)

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
                _remove_temp(temp_path)

    except WebSocketDisconnect:
        print(f"WS disconnected {session_id}")

    except Exception:
        traceback.print_exc()

    finally:
        print(f"WS closed {session_id}")


if __name__ == "__main__":
    import uvicorn

    ensure_temp_audio_dir()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
