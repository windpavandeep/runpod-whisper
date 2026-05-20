from faster_whisper import WhisperModel

model = WhisperModel(
        "distil-large-v3",
        device="cuda",
        compute_type="float16"
      )

print("Whisper loaded successfully")
