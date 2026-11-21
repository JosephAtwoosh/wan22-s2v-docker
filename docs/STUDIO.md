# Connecting the moviestudio

The studio already owns the two things a performance needs: a start still per
shot and (per shot) a dialogue track. This service converts those three inputs
into a clip. It does not touch the studio's database.

The mapping is deliberately boring:

| Studio | Performance folder | API |
|---|---|---|
| shot's start still | `start.jpg` | `start` file part |
| shot's dialogue WAV | `audio.wav` | `audio` file part |
| the beat text, rewritten as one paragraph | `prompt.txt` | `prompt` form field |
| one shot | one job | one job id |

The stop still is **not** sent. In an LTX-style keyframe pipeline it would be
the second conditioning frame; here it would be an end pose the audio did not
ask for, and that is exactly what made the acting stiff.

---

## One call

```bash
curl -s -X POST "$S2V_URL/v1/performances" \
  -H "Authorization: Bearer $S2V_KEY" \
  -F start=@shot-01/start.jpg \
  -F audio=@shot-01/line.wav \
  -F prompt="$(cat shot-01/prompt.txt)" \
  -F max_clips=1
```

The response is `202` with `id`, `status` and a `plan` block. Then:

```
GET /v1/jobs/{id}            until status == done | error
GET /v1/jobs/{id}/video      the mp4
GET /v1/jobs/{id}/manifest.json
```

A render is minutes, so the call is asynchronous by design. A PHP page that
blocks on it would hold the whole built-in server.

## PHP client

```php
final class S2V
{
    public function __construct(
        private string $base,
        private string $key,
        private int $timeout = 60,
    ) {}

    /** @return array{id:string,status:string,plan:array} */
    public function submit(string $still, string $audio, string $prompt, array $opts = []): array
    {
        $ch = curl_init($this->base . '/v1/performances');
        $fields = [
            'start'  => new CURLFile($still, 'image/jpeg', 'start.jpg'),
            'audio'  => new CURLFile($audio, 'audio/wav', 'audio.wav'),
            'prompt' => $prompt,
        ] + $opts; // size, seed, max_clips, conform_fps
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => $fields,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => $this->timeout,
            CURLOPT_HTTPHEADER     => ['Authorization: Bearer ' . $this->key],
        ]);
        $body = curl_exec($ch);
        $code = curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
        curl_close($ch);
        if ($code !== 202) {
            throw new RuntimeException("s2v submit failed: HTTP $code $body");
        }
        return json_decode($body, true, 512, JSON_THROW_ON_ERROR);
    }

    public function poll(string $id): array
    {
        // ... GET /v1/jobs/{id} until done/error
    }

    public function download(string $id, string $destination): void
    {
        // ... GET /v1/jobs/{id}/video  (Authorization header) -> file
    }
}
```

`store_media()` and the shot's media table are the studio's own; the service
returns a file, and the studio decides what to do with it.

## Polling from a CLI worker

The pattern that works with a single-threaded studio:

1. A studio action writes a row into a small `render_queue` table
   (`shot_id`, `status`, `job_id`, `error`, timestamps).
2. A CLI worker (`php tools/render_worker.php`) claims the oldest pending row,
   submits it, polls every 10 s, downloads the finished clip, calls
   `store_media()` + `shot_video_set()`, and marks the row done or failed.
3. The web UI only reads that table, so no page ever waits on the GPU.

This mirrors what the Qwen-TTS service already expects; see
`qwen-tts-docker/docs/STUDIO.md`.

## Prompts per shot

Write the prompt from the beat, in the order in
[docs/PERFORMANCE.md](PERFORMANCE.md): place and light, who and where, what the
audio is doing, one intention. Keep the previous shot's place nouns if the
scene is continuous — the model has no memory between jobs, so continuity has
to be in the text and in the still.

Do not reuse the still's image-generation prompt verbatim: image prompts are
written for a single frame (composition, lens, palette), performance prompts
are written for motion and intention.

## Retakes and provenance

- Keep `manifest.json` next to the clip. It has the input hashes, the seed and
  the resolved plan; a retake with the same seed and inputs is deterministic.
- A retake is `seed + 1`, not "regenerate and hope".
- If two clips of the same shot exist, the manifest is what tells them apart.

## 16 fps and the edit

Wan renders at 16 fps. If the timeline is 24 fps, pass `conform_fps=24` (or
`--conform-fps 24`): the renderer duplicates frames with ffmpeg so the clip is
natively 24 fps at the same duration and the same sync. Do not retime in the
editor per clip — one timeline-wide conform is easier to reason about than
twenty per-clip speed changes.

## Failure handling

| Failure | What the studio should do |
|---|---|
| `400` on submit | the folder is wrong (two images, no audio, empty prompt): fix the input, do not retry |
| `409` on video | still rendering; keep polling |
| `error` with `generate.py exited N` | read `/v1/jobs/{id}/log`; OOM means a smaller `size` or fewer clips |
| `error` with `no audio stream` | an infrastructure bug worth reporting: the clip was produced without sound and was rejected on purpose |
| container restarted mid-job | the job is marked `interrupted` on startup; resubmit |

The queue is one at a time. Two GPU renders at once will OOM the card, so do
not parallelise at the API; parallelise by rendering a scene's shots in a
`batch` on one process instead.
