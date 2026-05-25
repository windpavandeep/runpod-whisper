import asyncio
import json
import os
import struct
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn

# =========================
# CONFIG
# =========================

SERVER_VERSION = "2026-realtime-pcm"

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 3000

TEMP_AUDIO_DIR = "temp_audio"

os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

print(f"Loading Whisper model: {MODEL_NAME}")

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

print("Whisper model loaded.")

# =========================
# WHISPER OPTIONS
# =========================

WHISPER_OPTS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
}

# =========================
# PCM → WAV
# =========================


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
):

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


# =========================
# PARSE SOCKET FRAME
# =========================


def parse_pcm_frame(data: bytes):

    """
    Binary Frame Format:

    [4-byte chunkId][PCM int16 bytes]
    """

    if len(data) < 8:
        return None

    chunk_id = int.from_bytes(
        data[:4],
        byteorder="big",
        signed=False,
    )

    pcm_data = data[4:]

    if len(pcm_data) % 2 != 0:
        pcm_data = pcm_data[:-1]

    if len(pcm_data) < 2:
        return None

    return chunk_id, pcm_data


# =========================
# TRANSCRIBE
# =========================


def transcribe_pcm(
    pcm_bytes: bytes,
    chunk_id: int,
):

    if len(pcm_bytes) < 3200:
        return {
            "transcript": "",
            "segments": [],
        }

    wav_bytes = pcm_to_wav_bytes(pcm_bytes)

    temp_path = os.path.join(
        TEMP_AUDIO_DIR,
        f"{uuid.uuid4().hex}.wav"
    )

    try:

        with open(temp_path, "wb") as f:
            f.write(wav_bytes)

        print("TEMP FILE:", temp_path)
        print("FILE SIZE:", os.path.getsize(temp_path))

        segments_iter, info = model.transcribe(
            temp_path,
            **WHISPER_OPTS,
        )

        transcript_parts = []
        segments = []

        for index, seg in enumerate(segments_iter):

            text = seg.text.strip()

            if not text:
                continue

            transcript_parts.append(text)

            segments.append({
                "id": f"{chunk_id}-{index}",
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })

        return {
            "transcript": " ".join(transcript_parts),
            "segments": segments,
        }

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# SAFE SEND
# =========================


async def safe_send(
    websocket: WebSocket,
    payload: dict,
):

    try:
        await websocket.send_json(payload)
        return True

    except WebSocketDisconnect:
        return False


# =========================
# ROUTES
# =========================


@app.get("/")
async def root():

    return {
        "service": "printx-whisper",
        "version": SERVER_VERSION,
        "websocket": "/ws/transcribe",
    }


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


# =========================
# WEBSOCKET
# =========================


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    session_id = uuid.uuid4().hex[:8]

    print(f"WS CONNECTED: {session_id}")

    try:

        while True:

            message = await websocket.receive()

            # disconnect
            if message["type"] == "websocket.disconnect":
                break

            # metadata text
            if "text" in message and message["text"]:

                try:
                    meta = json.loads(message["text"])

                    if meta.get("type") == "end":
                        continue

                except:
                    pass

            # binary audio
            if "bytes" not in message:
                continue

            data = message["bytes"]

            parsed = parse_pcm_frame(data)

            if parsed is None:

                await safe_send(
                    websocket,
                    {
                        "error": "Invalid PCM frame",
                    }
                )

                continue

            chunk_id, pcm_bytes = parsed

            print(
                f"CHUNK {chunk_id} | PCM SIZE: {len(pcm_bytes)}"
            )

            try:

                result = await asyncio.to_thread(
                    transcribe_pcm,
                    pcm_bytes,
                    chunk_id,
                )

                success = await safe_send(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "transcript": result["transcript"],
                        "segments": result["segments"],
                    }
                )

                if not success:
                    break

            except Exception as e:

                traceback.print_exc()

                success = await safe_send(
                    websocket,
                    {
                        "chunkId": chunk_id,
                        "error": str(e),
                    }
                )

                if not success:
                    break

    except WebSocketDisconnect:

        print(f"WS DISCONNECTED: {session_id}")

    finally:

        print(f"WS CLOSED: {session_id}")


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )