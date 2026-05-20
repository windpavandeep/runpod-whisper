import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8",
)


@dataclass
class RoomState:
    speaker: WebSocket | None = None
    viewers: set[WebSocket] = field(default_factory=set)


rooms: dict[str, RoomState] = {}
rooms_lock = asyncio.Lock()


async def get_or_create_room(room_id: str) -> RoomState:
    async with rooms_lock:
        if room_id not in rooms:
            rooms[room_id] = RoomState()
        return rooms[room_id]


async def remove_empty_room(room_id: str) -> None:
    async with rooms_lock:
        room = rooms.get(room_id)
        if room and room.speaker is None and not room.viewers:
            del rooms[room_id]


async def broadcast_to_room(room_id: str, payload: dict) -> None:
    room = rooms.get(room_id)
    if not room:
        return

    targets = list(room.viewers)
    if room.speaker:
        targets.append(room.speaker)

    for ws in targets:
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in room.viewers:
                room.viewers.discard(ws)


async def transcribe_webm(audio_bytes: bytes) -> str:
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        segments, _info = model.transcribe(
            temp_path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        return "".join(segment.text for segment in segments).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/rooms")
async def create_room():
    room_id = str(uuid.uuid4())[:8]
    await get_or_create_room(room_id)
    return {"room_id": room_id}


@app.websocket("/ws/room/{room_id}")
async def room_websocket(websocket: WebSocket, room_id: str):
    role = websocket.query_params.get("role", "viewer")

    if role not in ("speaker", "viewer"):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    room = await get_or_create_room(room_id)

    try:
        if role == "speaker":
            if room.speaker is not None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Room already has a speaker",
                })
                await websocket.close(code=1008)
                return

            room.speaker = websocket
            await broadcast_to_room(room_id, {
                "type": "status",
                "speaker_connected": True,
            })

            while True:
                audio_bytes = await websocket.receive_bytes()

                if not audio_bytes:
                    continue

                try:
                    text = await transcribe_webm(audio_bytes)
                    await broadcast_to_room(room_id, {
                        "type": "transcript",
                        "text": text,
                    })
                except Exception as e:
                    print("Transcription error:", e)
                    await websocket.send_json({
                        "type": "transcript",
                        "text": "",
                        "error": str(e),
                    })

        else:
            room.viewers.add(websocket)
            await websocket.send_json({
                "type": "status",
                "speaker_connected": room.speaker is not None,
            })

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("Room connection error:", e)
    finally:
        if role == "speaker" and room.speaker is websocket:
            room.speaker = None
            await broadcast_to_room(room_id, {
                "type": "status",
                "speaker_connected": False,
            })
        elif role == "viewer" and websocket in room.viewers:
            room.viewers.discard(websocket)

        await remove_empty_room(room_id)
        print(f"Client disconnected from room {room_id} ({role})")


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("Client connected (solo)")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            if not audio_bytes:
                continue

            try:
                text = await transcribe_webm(audio_bytes)
                await websocket.send_json({"text": text})
            except Exception as e:
                print("Transcription error:", e)
                await websocket.send_json({"text": "", "error": str(e)})

    except Exception as e:
        print("Connection error:", e)
    finally:
        print("Client disconnected (solo)")
