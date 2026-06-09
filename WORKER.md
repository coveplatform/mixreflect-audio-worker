# djmix audio worker — DSP grounding for mixreflect score reports

`worker.py` exposes the djmix analysis core (`analyze.py`) as an HTTP service that
takes a **track URL** and returns real measured audio features. It's what fills
mixreflect's `AUDIO_WORKER_URL` hook so the "audio-grounded" score report is
actually grounded in the audio instead of guessing from the title + genre.

## What it returns

`POST /analyze {"url": "..."}` →

```json
{
  "features": {
    "durationSec": 212.0,
    "tempo": 128.0,
    "key": "F# minor",
    "loudnessLufs": -9.4,
    "energy": 0.7,
    "spectral": { "sub": 0.17, "bass": 0.17, "lowMid": 0.27, "mid": 0.27, "high": 0.12 },
    "introLiftSec": 14.0,
    "energyDips": [{ "startSec": 70.0, "endSec": 92.0, "dropDb": 8.4 }],
    "sections": [{ "kind": "drop", "startSec": 14.0, "endSec": 70.0 }, ...],
    "vocalPresence": 0.35,
    "gridConfidence": "high"
  },
  "took": 6.2
}
```

`features` is `null` when the track couldn't be fetched/analyzed (bad link,
download blocked, unsupported source) — mixreflect then falls back to its
non-grounded read. This shape matches `AudioFeatures` in
`mixreflect/src/lib/audio-analysis.ts`; `worker.to_audio_features()` does the
mapping from an `analyze.analyze_file()` record.

## How it gets the audio

1. **Direct audio links** (`.mp3 .wav .flac .m4a .ogg .opus`) → downloaded directly.
2. **SoundCloud / YouTube / Bandcamp / etc.** → `yt-dlp`.

Either way `ffmpeg` transcodes to a mono 22.05 kHz wav that libsndfile can read,
`analyze.py` measures it, and the temp audio is deleted immediately after. Only
the derived numbers leave the box. There's a 60 MB download cap and a 45s fetch
timeout. Demucs/stems are force-disabled (`DJMIX_STEMS=0`) — far too heavy for a
request-time worker; `analyze.py` uses its loudness-based structure fallback.

## Run / deploy

```bash
pip install -r requirements-worker.txt   # numpy soundfile mutagen yt-dlp
# ffmpeg must be installed and on PATH
AUDIO_WORKER_SECRET=some-shared-secret python worker.py --port 8090
```

- `--port` / `$PORT` (default 8090), `--host` / `$HOST` (default `0.0.0.0`).
- `AUDIO_WORKER_SECRET` — shared bearer token. If unset, `/analyze` is **open**;
  only run it that way on a private network.
- Needs a normal (non-serverless) host: a small VM / container / Fly/Render/Railway
  box. numpy + ffmpeg + yt-dlp won't run in Next's serverless functions — which is
  exactly why this is a separate service.

A `Dockerfile` is the natural packaging step (base image + `apt-get install ffmpeg`
+ `pip install -r requirements-worker.txt`).

## Wire it into mixreflect

Set on the mixreflect deployment:

```
AUDIO_WORKER_URL=https://<where-this-runs>
AUDIO_WORKER_SECRET=some-shared-secret      # same value as above
```

`acquireAudioFeatures()` already routes non-Spotify links (SoundCloud/YouTube/
Bandcamp/direct files) to `${AUDIO_WORKER_URL}/analyze`, and `generateReport()`
feeds the measured numbers to the model as ground truth. No mixreflect code
change is needed beyond the env vars.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | liveness probe (`stems`, `stemBackend`) |
| POST | `/analyze` | bearer | `{url, deep?}` → `{features, took}` |

`deep: true` runs the stem-separation pass (drums/bass/vocals/other balance +
richer structure); the default fast read skips it. mixreflect sends `deep: true`
only for paid "deep" reports.

## Stems via Replicate (no torch on this box)

Demucs is heavy (torch, ideally a GPU). Instead of installing it here, the worker
offloads the GPU separation to **Replicate** and runs the light numpy structure
analysis locally. To enable, set on the deployment:

```
REPLICATE_API_TOKEN=r8_xxx                  # from replicate.com/account/api-tokens
# optional — pin the demucs model+version (verify the SHA on the model page):
REPLICATE_DEMUCS_MODEL=cjwbw/demucs:<version>
```

- `stems.available()` is true when a local demucs install exists **or** the token
  is set; `separate()` prefers local demucs, else calls Replicate.
- Cost: ~$0.02 per separated track, and only `deep` requests pay it (~paid reports).
- Verify: `GET /health` → `"stemBackend": "replicate"` once the token is live.
- No token, no torch → stems stay off and the worker uses the loudness-based
  structure fallback (still grounded, just no per-stem detail).
