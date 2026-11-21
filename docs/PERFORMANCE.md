# Writing a performance

A performance is judged on three things, in this order: **the track**, **the
still**, and **the prompt**. The model is conditioned on all three, and the
weakest one sets the result.

---

## The three files

```
shot-01/
    start.jpg     1920x1080 (or larger), the first frame of the shot
    audio.wav     16 kHz+ mono; one speaker, no music
    prompt.txt    one paragraph
```

`plan` will tell you the render resolution before you spend a GPU-minute:

```bash
wan22-s2v plan performances/shot-01
```

### The still

- **One image.** Two images are rejected by name. The model has one reference
  channel; a second still only makes the folder ambiguous.
- **Start frame, not a master shot.** The still is frame 0. Whatever pose,
  eyeline and framing it has, the clip inherits. A full-body still pulls the
  camera back; a close-up keeps it close.
- **No EXIF surprises.** Upstream ignores the orientation tag, so a phone photo
  that relies on it will render sideways. Rotate the file itself.
- **Big enough.** Wan never upscales. A 320x240 still renders at 320x256 and
  the preset's area is wasted; `plan` warns when the still is under half the
  target area.
- **Neutral, evenly lit, on-model.** Every reference the model sees comes from
  this one frame. If the character's design lives in three other shots, this
  still has to carry it.

### The track

- **Speech, one voice, no bed.** The audio encoder (wav2vec2) is being asked to
  drive lips and gesture. Music, room tone and overlapping speakers all become
  mouth movement and random head turns.
- **No clipping, no long silence at the top.** The model starts acting at
  frame 0; two seconds of silence is two seconds of the model improvising.
  Trim the lead-in, or leave it deliberately short.
- **Length is free-form.** The clip count is computed from the duration; 42.5 s
  is nine clips in one process. Over 300 s the API refuses by default
  (`MAX_AUDIO_SECONDS`) — split the shot instead, so a failure costs one take.
- **A whisper is not quiet in the data.** Normalise loudness across shots; a
  track 10 dB quieter than the last one changes how much motion the model
  infers.

### The prompt

One paragraph, 20–4000 characters, whitespace collapsed. It is not a shot
description for a storyboard; it is the world the model has to build around
the speaker.

Write it in this order:

1. **Place, light, props.** "A grey dawn in the farmyard, frost on the cobbles,
   the coop door half open, straw scattered by the step."
2. **Who and where they are.** "The cockerel stands on the mounting block,
   facing the coop."
3. **What the audio is doing.** "He turns his head to speak, chest out,
   impatient."
4. **One intention, not a pose.** "He is about to interrupt."

Do not write "he raises his left wing to 45 degrees at the end". A pose is a
constraint the audio did not ask for; that is the `stop.jpg` problem in words.
Describe what the character wants and let the track time the delivery.

Good:

> A grey dawn in the farmyard, frost on the cobbles, the coop door half open.
> The cockerel stands on the mounting block, chest out, and turns to the coop
> to speak, impatient, as if he has been waiting.

Weak:

> A chicken. Close up. Talking. Happy. 4k, cinematic, highly detailed.

The second one has no place, so the model invents one per frame; and "happy"
fights the line being delivered.

---

## What the model does with them

| Input | Effect |
|---|---|
| still | identity, costume, framing, background, lighting |
| audio | mouth shapes, blinks, head motion, gesture timing, tempo |
| prompt | scene nouns, action verbs, mood words, camera behaviour |
| seed | everything else — the only honest way to get a different take |

Same still + same audio + same prompt + same seed is a deterministic render
(the upstream pipeline is). Change one variable at a time.

---

## Retakes

1. **Different seed** is the cheapest retake: same inputs, different
   performance. `--seed 12345` or `--seed -1` and read the seed back from the
   manifest.
2. **Tighten the prompt** if the motion is right but the scene is wrong: fix
   nouns, not adjectives.
3. **Fix the still** if identity or framing is wrong. No prompt repairs a bad
   reference.
4. **Fix the audio** if timing is wrong. No prompt repairs a bad track.

Keep the manifest with the clip. It records the input hashes, the resolved
parameters and the seed, which is what makes a retake reproducible six months
later.

---

## Cost intuition

Cost is proportional to clips, and clips are
`ceil(audio_seconds * 16 / 80)`. A five-second line is one clip; a ten-second
line is two; a 42-second monologue is nine. Model load is paid once per
process, so a batch of short performances on one card is much cheaper per clip
than nine separate `perform` calls.

`plan` prints the clip count without touching the GPU. Use it on the whole
scene before renting anything:

```bash
wan22-s2v batch performances/scene-01 --dry-run
```
