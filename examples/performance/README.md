# Example performance

This folder shows the shape of a performance, not a ready-to-render one: the
repository ships no images or audio, because they are your material.

```text
performances/shot-01/
    start.jpg        # or .png/.webp: the first frame
    audio.wav        # or .mp3/.flac/.m4a: the track
    prompt.txt       # one paragraph, see ../prompt.txt
    performance.json # optional overrides
```

A minimal `performance.json`:

```json
{
  "seed": 42,
  "size": "720p",
  "max_clips": 1
}
```

Check it without a GPU:

```bash
wan22-s2v plan performances/shot-01
```

Render it:

```bash
wan22-s2v perform performances/shot-01 --follow
```
