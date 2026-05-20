from faster_whisper import WhisperModel

model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8"
)

print("Whisper Loaded Successfully")