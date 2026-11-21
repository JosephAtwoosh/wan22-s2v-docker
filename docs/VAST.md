# Renting a GPU for this

The weights do not fit in the image, so the sequence is:

```
CI builds image -> GHCR -> Vast instance -> download weights once -> doctor ->
render -> destroy
```

Nothing below has been run yet on a card. This is the checklist to follow for
the **first** run, whose purpose is verification: one shot, measured, then
destroy.

## 1. Make the image pullable

The build workflow pushes `ghcr.io/<owner>/wan22-s2v-docker:latest`. GHCR
packages are private even in a public repository:

> Package page → *Package settings* → *Change visibility* → **Public**.

Without that, Vast's pull fails with `unauthorized`. Alternatively give the
instance a registry token in the template's docker-login fields.

## 2. Choose the card

Every offer has to clear all six requirements, not just VRAM:

| Requirement | Value | Why |
|---|---|---|
| VRAM | **80 GB** (48 GB marginal, untested) | DiT shards are 32.6 GB and go on the card for every clip |
| Host RAM | **64 GB** | offloaded DiT (32.6 GB) + text encoder (11.4 GB) resident together |
| Disk | **150 GB** | 15 GB image + 45.6 GB weights + outputs |
| Driver | **570+** | CUDA 12.8 |
| Download | **< $4/TB** | weights are a 45.6 GB pull |
| Upload | **< $4/TB** | outputs are small, but a bad offer charges either way |

`tools/vast.js search` checks all of them locally and refuses to rent an offer
that misses one. Vast also lists GB10 (Grace Blackwell) boxes at ~$0.27/h that
appear to clear the VRAM bar; they are ARM with unified memory, and this image
is amd64 with a cp312 x86_64 flash-attn wheel, so they cannot run it.

| Card | VRAM | Rough price | Fit |
|---|---|---|---|
| A100 SXM4 80 GB | 80 GB | $1.00–1.60/h | the upstream-recommended configuration |
| H100 PCIE 80 GB | 80 GB | $1.80–2.60/h | faster, same fit |
| L40S 48 GB | 48 GB | $0.70–1.00/h | short 480p clips only; 42.6 GiB peak measured, ~1 GiB headroom |

The checkpoint's DiT shards are **32.6 GB** in bf16 and are moved onto the card
for every clip; the UMT5 encoder (11.4 GB) is also on the card while the prompt
is encoded. Measured peaks: **42.6 GiB** at 576x576/48 frames, **49.3 GiB** at
768x768/80 frames. A 48 GB card (~44 GiB usable) therefore cannot run the
default 80-frame clip, and only just fits the short 480p one. Below 40 GB the
weights do not fit at all. `doctor` encodes exactly these thresholds.

Host RAM is the other half of `--offload_model`: the DiT (32.6 GB) and the UMT5
encoder (11.4 GB) both sit in host memory between clips. 64 GB is the minimum,
96 GB is comfortable. Vast's `ram_gb` column in `search` is the number to read.

The image pulls in about a minute or two on a hosted box: 24 layers, ~7.7 GB
compressed. That transfer is counted too — budget it with the weights.

`tools/vast.js` filters by VRAM in **GB**, not MB (Vast's own API takes GB; the
MB mistake returns an empty market):

```bash
node tools/vast.js search "A100 SXM4,H100 PCIE" 79000 150
```

## 3. Create the instance

```bash
vastai create instance <offer_id> \
  ghcr.io/josephatwoosh/wan22-s2v-docker:latest \
  --disk 150 \
  --ssh --direct \
  --label wan22-s2v-verify \
  --onstart-cmd 'wan22-s2v serve' \
  --env '-p 8000:8000 -e WAN_API_KEY=<secret>'
```

- **150 GB disk**: 45.6 GB weights + outputs + the image itself.
- `--onstart-cmd 'wan22-s2v serve'`: the image's default command is `doctor`,
  which exits; the service has to be the long-running process.
- `-p 8000:8000` maps the API. Vast shows the public URL/port on the instance
  page.

Weights, once the instance is up:

```bash
vastai ssh <instance>          # or the ssh command on the instance page
wan22-s2v download             # ~45.6 GB, public repo, ~5-15 min
wan22-s2v doctor               # must be READY before rendering
```

If `doctor` fails on VRAM, stop: fix the card choice rather than the code.

## 4. Verify, measured

First, the plumbing, with no assets at all. `selftest` paints its own still and
synthesises a three-second tone, then renders one 48-frame clip at 480p:

```bash
wan22-s2v selftest
```

It proves the weights load, attention runs, the VAE decodes and the mux works.
If it fails, the log names the layer. It does not prove likeness — that is what
the real shot is for.

Then one real performance, deliberately small:

```bash
wan22-s2v plan    /performances/shot-01 --size 480p --max-clips 1
wan22-s2v perform /performances/shot-01 --size 480p --max-clips 1 --seed 42 --follow
```

Record from `manifest.json`:

- `timing.render_seconds` and `timing.realtime_factor`
- `output.duration`, `output.width`, `output.height`
- `environment.gpu`, `environment.vram_gib`

Then a full-size run (`--size 720p`, no clip cap) to see the real cost. Only
after both should README's Performance section claim numbers.

## 5. Cost, in round numbers

Weight download is the largest single transfer (~$0.20 at $4/TB). A 10-second
clip at 720p is two clips of generation; on an A100 that is minutes, not
seconds, because 40 sampling steps run per clip. Model load (once per process)
is a large fraction of a one-clip job — which is why `batch` renders a scene in
one process and `perform` should be used for retakes.

Budget **$1–3** for the verification session: one hour of an A100 plus
transfer. Destroy the instance when done; stopped instances still bill disk.

## 6. Destroy

```bash
vastai destroy instance <instance>
```

`tools/vast.js destroy` refuses without `--yes`, for the same reason.
