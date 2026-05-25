# Temp audio folder (RunPod / VPS)

Avoid `/tmp` for realtime WAV files — use a dedicated project folder instead.

## One-time setup on server

```bash
cd /root/runpod-whisper   # or your deploy path
mkdir -p temp_audio
chmod -R 777 temp_audio
ls -ld temp_audio
ls -ld /tmp   # should show drwxrwxrwt (sticky "t")
```

Optional explicit env (systemd / PM2):

```bash
export WHISPER_TEMP_AUDIO_DIR=/root/runpod-whisper/temp_audio
```

Default without env: `{repo}/runpod-whisper/temp_audio` next to `server.py`.

## Restart service

```bash
sudo systemctl restart whisper
journalctl -u whisper -f
```

## What to look for in logs

On startup:

```
TEMP_AUDIO_DIR: /root/runpod-whisper/temp_audio (writable: True)
```

Per chunk:

```
TEMP PATH (chunk 1): /root/runpod-whisper/temp_audio/abc123.wav
FILE SIZE (chunk 1): 96044
```

| File size | Meaning |
|-----------|---------|
| 0–500 | Broken frontend / invalid PCM |
| 50,000+ | Valid ~3s 16 kHz WAV |

## Health check

```bash
curl http://YOUR_HOST:8000/health
```

Expect `"version": "2026-05-26-temp-audio-dir"` and `"temp_dir_writable": true`.
