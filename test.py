from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5
)

print("Detected language:", info.language)

for segment in segments:
    print(segment.text)
