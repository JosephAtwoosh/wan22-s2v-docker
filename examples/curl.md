# curl examples

The API takes multipart: two files and a prompt. Every response is JSON except
the video itself.

```bash
export S2V=http://localhost:8000
export KEY=change-me
```

## Submit

```bash
curl -s -X POST "$S2V/v1/performances" \
  -H "Authorization: Bearer $KEY" \
  -F start=@performances/shot-01/start.jpg \
  -F audio=@performances/shot-01/audio.wav \
  -F prompt="$(cat performances/shot-01/prompt.txt)" \
  -F max_clips=1 | jq
```

```json
{
  "id": "20261121-064000-a1b2c3",
  "status": "queued",
  "created_at": "2026-11-21T06:40:00+00:00",
  "plan": {
    "out_width": 1280,
    "out_height": 704,
    "clips": 1,
    "frames": 80,
    "video_seconds": 5.0,
    "audio_seconds": 4.99,
    "warnings": []
  }
}
```

`max_clips` is a preview: it caps the clip count and the tail of the audio is
cut. Omit it for the full performance.

Optional parts, if you want to pin parameters for this job:
`size=480p`, `seed=42`, `start_from_ref=true`, `conform_fps=24`, or a
`prompt_file` instead of the `prompt` field.

## Poll

```bash
JOB=20261121-064000-a1b2c3
watch -n 5 "curl -s $S2V/v1/jobs/$JOB -H 'Authorization: Bearer $KEY' | jq '{status, error, timing}'"
```

`status` is `queued`, `running`, `done` or `error`. When done, `manifest` holds
the full record.

## Download

```bash
curl -s "$S2V/v1/jobs/$JOB/video" -H "Authorization: Bearer $KEY" -o clip.mp4
curl -s "$S2V/v1/jobs/$JOB/manifest.json" -H "Authorization: Bearer $KEY" -o clip.json
curl -s "$S2V/v1/jobs/$JOB/log" -H "Authorization: Bearer $KEY" | tail -40
```

A browser `<video>` tag cannot send headers; use the token query parameter for
that one case:

```html
<video src="http://gpu-box:8000/v1/jobs/20261121-064000-a1b2c3/video?token=change-me" controls></video>
```

## Health and queue

```bash
curl -s "$S2V/health" | jq '{status, gpu, vram_gib, engine_ready, running_jobs}'
curl -s "$S2V/v1/jobs?limit=5" -H "Authorization: Bearer $KEY" | jq '.jobs[].status'
```

`/health` is open so a load balancer or the Vast template can probe it without
a key. It reports `degraded` when the engine is not ready (missing weights or
checkout), and `engine_detail` says which.

## Errors

| Code | Meaning |
|---|---|
| `400` | the folder is invalid: two images, no audio, empty prompt, bad extension, over a limit |
| `401` | missing/incorrect key |
| `409` | `video` requested before the job finished |
| `413` | upload larger than `MAX_UPLOAD_MB` |
| `500` | a render failed; `error` on the job and `/log` say why |
