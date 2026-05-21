import os
import uuid
import subprocess

from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper model loaded!")

TEMP_DIR = "temp_audio"

os.makedirs(TEMP_DIR, exist_ok=True)


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("Client connected")

    audio_chunks = []

    try:

        while True:

            message = await websocket.receive()

            if "text" in message:

                if message["text"] == "END":

                    webm_path = f"{TEMP_DIR}/{uuid.uuid4()}.webm"

                    wav_path = f"{TEMP_DIR}/{uuid.uuid4()}.wav"

                    with open(webm_path, "wb") as f:
                        for chunk in audio_chunks:
                            f.write(chunk)

                    command = [
                        "ffmpeg",
                        "-i",
                        webm_path,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        wav_path,
                        "-y"
                    ]

                    subprocess.run(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )

                    segments, info = model.transcribe(
                        wav_path,
                        beam_size=5
                    )

                    transcript = ""

                    for segment in segments:
                        transcript += segment.text + " "

                    await websocket.send_json({
                        "text": transcript.strip(),
                        "language": info.language
                    })

                    os.remove(webm_path)
                    os.remove(wav_path)

                    audio_chunks = []

            elif "bytes" in message:

                audio_chunks.append(message["bytes"])

    except Exception as e:

        print("ERROR:", e)

    finally:

        print("Client disconnected")
