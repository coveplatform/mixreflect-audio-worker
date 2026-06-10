"""djmix audio worker — URL → real DSP features, for mixreflect's score reports.

mixreflect (`src/lib/audio-analysis.ts`) already knows how to call an external
analyzer: it POSTs `{ "url": "<track link>" }` to `${AUDIO_WORKER_URL}/analyze`
with an optional `Authorization: Bearer ${AUDIO_WORKER_SECRET}` and spreads the
JSON it gets back into its `AudioFeatures` shape. Nothing was filling that hook,
so the report's "audio-grounded" read was running blind.

This service fills it. It reuses the djmix analysis core (`analyze.py` — the same
BPM / key / energy / spectral / structure engine the DJ tool uses) but fronts it
with a downloader so it works on a *link* instead of a local file:

    1. direct audio link (.mp3/.wav/.flac/.m4a/.ogg) → downloaded directly
    2. SoundCloud / YouTube / Bandcamp / etc.        → yt-dlp
    …then ffmpeg transcodes whatever came down to a mono wav that libsndfile can
    read, `analyze.analyze_file` measures it, and `to_audio_features` maps the
    rich djmix record onto mixreflect's `AudioFeatures` contract.

Audio is fetched to a temp dir and deleted right after analysis — only the
derived numbers (tempo, key, energy, the section/energy arc) are returned.

Run it:
    pip install -r requirements-worker.txt      # numpy soundfile mutagen yt-dlp
    # ffmpeg must be on PATH
    AUDIO_WORKER_SECRET=some-shared-secret python worker.py --port 8090

Then on mixreflect set:
    AUDIO_WORKER_URL=https://<where-this-runs>
    AUDIO_WORKER_SECRET=some-shared-secret      # same value

Endpoints:
    GET  /health                 liveness (no auth)
    POST /analyze  {url}         bearer-auth; returns AudioFeatures JSON
"""
import argparse
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import analyze  # the shared DSP core (BPM/key/energy/spectral/structure)

# Stem separation (demucs/torch) reads real verse/chorus/drop structure even on
# loud, consistently-mastered tracks — where the loudness-based detector collapses
# the whole song into one flat section and makes polished hits look monotonous.
# Heavy: needs torch + demucs installed (ideally a GPU box). Set DJMIX_STEMS=0 to
# force it off and fall back to the lighter loudness structure.
os.environ.setdefault("DJMIX_STEMS", "1")

DIRECT_AUDIO_EXT = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".opus")
MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024  # 60 MB cap — a single track, not an album rip
DOWNLOAD_TIMEOUT = 45  # seconds for the network fetch
# Only analyse the first N seconds — the analysis arrays scale with duration, and
# this keeps peak memory inside a small (512 MB) box. Plenty to score a track.
MAX_ANALYZE_SECS = int(os.environ.get("MAX_ANALYZE_SECS", "210"))
# Serialize analysis so concurrent uploads can't run in parallel and double the
# memory (→ OOM on a small box). Excess requests wait briefly, then shed load
# (return no features → caller falls back to a non-grounded read, never a crash).
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "1"))
QUEUE_WAIT_SECS = int(os.environ.get("QUEUE_WAIT_SECS", "20"))
_ANALYZE_SEM = threading.BoundedSemaphore(MAX_CONCURRENCY)


# ── download ─────────────────────────────────────────────────────────────────

def _is_direct_audio(url: str) -> bool:
    return urlparse(url).path.lower().endswith(DIRECT_AUDIO_EXT)


def _fetch_direct(url: str, dest: str) -> bool:
    """Stream a direct audio URL to `dest`, capped. Returns True on success."""
    req = Request(url, headers={"User-Agent": "djmix-worker/1.0"})
    try:
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            total = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("file exceeds size cap")
                    f.write(chunk)
        return total > 0
    except Exception as e:  # noqa: BLE001
        print(f"[worker] direct fetch failed: {e}", flush=True)
        return False


_COOKIE_PATH: str | None = None

def _cookie_file() -> str | None:
    """Resolve a yt-dlp cookies file: base64 env (YTDLP_COOKIES_B64) → temp file,
    or an explicit path (YTDLP_COOKIES_FILE). Cached after first materialisation."""
    global _COOKIE_PATH
    if _COOKIE_PATH and os.path.exists(_COOKIE_PATH):
        return _COOKIE_PATH
    b64 = os.environ.get("YTDLP_COOKIES_B64")
    if b64:
        try:
            import base64  # noqa: PLC0415
            path = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            _COOKIE_PATH = path
            return path
        except Exception as e:  # noqa: BLE001
            print(f"[worker] failed to load YTDLP_COOKIES_B64: {e}", flush=True)
    p = os.environ.get("YTDLP_COOKIES_FILE")
    if p and os.path.exists(p):
        _COOKIE_PATH = p
        return p
    return None


def _fetch_ytdlp(url: str, out_dir: str) -> str | None:
    """Download bestaudio via yt-dlp and transcode to wav. Returns the wav path."""
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError:
        print("[worker] yt-dlp not installed — streaming links unsupported", flush=True)
        return None

    out_tmpl = os.path.join(out_dir, "src.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "max_filesize": MAX_DOWNLOAD_BYTES,
        "socket_timeout": DOWNLOAD_TIMEOUT,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
    }
    # Cookies are the real fix for YouTube bot-walls on cloud IPs (base64 env
    # YTDLP_COOKIES_B64 or a path YTDLP_COOKIES_FILE).
    # A residential proxy is the real fix for YouTube's datacenter-IP block —
    # the IP is the problem, not auth. Set YTDLP_PROXY to a proxy URL.
    proxy = os.environ.get("YTDLP_PROXY")
    if proxy:
        opts["proxy"] = proxy

    cookies = _cookie_file()
    if cookies:
        opts["cookiefile"] = cookies
    # NB: do NOT pin player_client — forcing android/ios excludes the client that
    # actually has downloadable formats ("Requested format is not available").
    # yt-dlp keeps its default client selection current; with a residential proxy
    # the defaults download fine.
    # Only pull the first MAX_ANALYZE_SECS so the wav (and analysis) stays small.
    try:
        from yt_dlp.utils import download_range_func  # noqa: PLC0415
        opts["download_ranges"] = download_range_func(None, [(0, MAX_ANALYZE_SECS)])
        opts["force_keyframes_at_cuts"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # extract_info(download=True) == download, but also hands us the
            # metadata — `duration` is the FULL source length even when
            # download_ranges only pulled the first MAX_ANALYZE_SECS.
            info = ydl.extract_info(url, download=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Geo-blocked videos surface as "Video unavailable" from the proxy's
        # exit region while playing fine elsewhere (observed live: a region-
        # locked music video). One retry WITHOUT the proxy — the datacenter IP
        # often passes geo even though it needs the proxy for bot-walls.
        if proxy and "unavailable" in msg.lower():
            print("[worker] unavailable via proxy — retrying direct", flush=True)
            opts.pop("proxy", None)
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except Exception as e2:  # noqa: BLE001
                print(f"[worker] yt-dlp failed (direct retry): {e2}", flush=True)
                return None
        else:
            print(f"[worker] yt-dlp failed: {e}", flush=True)
            return None

    src_dur = None
    try:
        src_dur = float(info.get("duration")) if info and info.get("duration") else None
    except Exception:  # noqa: BLE001
        pass
    for nm in os.listdir(out_dir):
        if nm.lower().endswith(".wav"):
            return os.path.join(out_dir, nm), src_dur
    return None


def _soundfile_readable(path: str) -> bool:
    """True if libsndfile can open it directly (wav/flac/ogg/aiff) — then we can
    skip the ffmpeg transcode entirely."""
    try:
        import soundfile as sf  # noqa: PLC0415
        with sf.SoundFile(path):
            return True
    except Exception:  # noqa: BLE001
        return False


def _transcode_to_wav(src: str, dest: str) -> bool:
    """ffmpeg → mono 16 kHz wav. Everything the analysis measures (bands split
    at 160 Hz / 2 kHz, tempo, key, structure) lives comfortably under the 8 kHz
    Nyquist, and vs 22.05 kHz it's ~27% fewer samples through every FFT — less
    memory AND less CPU on the small box."""
    if not shutil.which("ffmpeg"):
        print("[worker] ffmpeg not on PATH — cannot transcode", flush=True)
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-t", str(MAX_ANALYZE_SECS), "-ac", "1", "-ar", "16000", dest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return os.path.exists(dest) and os.path.getsize(dest) > 0
    except Exception as e:  # noqa: BLE001
        print(f"[worker] transcode failed: {e}", flush=True)
        return False


def _probe_duration(path: str) -> float | None:
    """Source duration via ffprobe (pre-cap), so we can report how much of a
    long track the MAX_ANALYZE_SECS window actually covered."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:  # noqa: BLE001
        return None


def acquire_wav(url: str, work_dir: str) -> tuple[str, float | None] | None:
    """Get a libsndfile-readable wav for `url` → (path, source duration secs or
    None). The wav is capped at MAX_ANALYZE_SECS; the duration is the FULL
    source length when we can know it, so the app can tell a truncated read."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return None

    if _is_direct_audio(url):
        raw = os.path.join(work_dir, "raw_input")
        if not _fetch_direct(url, raw):
            return None
        src_dur = _probe_duration(raw)
        # Always transcode → caps duration (MAX_ANALYZE_SECS) and normalises to
        # mono 22.05 kHz, so even a long wav/flac upload stays inside memory.
        wav = os.path.join(work_dir, "track.wav")
        if _transcode_to_wav(raw, wav):
            return wav, src_dur
        # Fallback: ffmpeg couldn't handle it but libsndfile might (uncapped).
        return (raw, src_dur) if _soundfile_readable(raw) else None

    # streaming site → yt-dlp produces a FULL-quality wav (48 kHz stereo, ~40 MB);
    # transcode it down to mono 22.05 kHz too, or analyze.py OOMs the small box.
    yt = _fetch_ytdlp(url, work_dir)
    if not yt:
        return None
    yt_path, src_dur = yt
    small = os.path.join(work_dir, "track.wav")
    return (small, src_dur) if _transcode_to_wav(yt_path, small) else (yt_path, src_dur)


# ── map djmix record → mixreflect AudioFeatures ──────────────────────────────

def _format_key(k: str | None) -> str | None:
    """'F#m' → 'F# minor', 'F#' → 'F# major', '' → None."""
    if not k:
        return None
    k = k.strip()
    if k.endswith("m") and not k.endswith("aj"):
        return f"{k[:-1]} minor"
    return f"{k} major"


def _derive_spectral(efeat: dict | None) -> dict | None:
    """Approximate a 5-band balance (sums ~1) from the loudness-independent cues
    analyze.py keeps: `sub` (low-band proportion) and `bright` (high-band)."""
    if not efeat:
        return None
    low = max(0.0, min(1.0, float(efeat.get("sub", 0.0))))
    high = max(0.0, min(1.0, float(efeat.get("bright", 0.0))))
    mid = max(0.0, 1.0 - low - high)
    return {
        "sub": round(low * 0.5, 3),
        "bass": round(low * 0.5, 3),
        "lowMid": round(mid * 0.5, 3),
        "mid": round(mid * 0.5, 3),
        "high": round(high, 3),
    }


def _derive_dips(structure: dict | None) -> list:
    """Energy dips = the sections where the track pulls back (breakdowns / quiet
    interior sections), expressed relative to the fullest section."""
    if not structure:
        return []
    secs = structure.get("sections") or []
    if not secs:
        return []
    ref = max((s.get("energy", 0.0) or 0.0 for s in secs), default=0.0) or 1.0
    dips = []
    for s in secs:
        kind = s.get("kind")
        if kind in ("intro", "outro"):
            continue
        e = s.get("energy", 0.0) or 0.0
        if e <= 0:
            continue
        drop_db = 20.0 * math.log10(ref / max(e, 1e-3))
        # only count a genuine pull-back, not normal section-to-section variation
        if kind == "breakdown" or drop_db >= 4.0:
            dips.append({
                "startSec": s.get("start"),
                "endSec": s.get("end"),
                "dropDb": round(max(0.0, drop_db), 1),
            })
    dips.sort(key=lambda d: d["dropDb"], reverse=True)
    return dips[:4]


def _sections_summary(structure: dict | None) -> list:
    if not structure:
        return []
    return [
        {"kind": s.get("kind"), "startSec": s.get("start"), "endSec": s.get("end")}
        for s in (structure.get("sections") or [])
    ]


def _mean(arr):
    return round(sum(arr) / len(arr), 3) if arr else None


def _acoustic_fingerprint(rec: dict, spectral: dict | None, energy: float | None) -> dict:
    """A compact, link-independent identity descriptor for a track. Two uploads of
    the same song (even via different links / re-encodes) land close on these;
    mixreflect compares them to recognise re-uploads and new versions. Uses only
    cues that survive re-encoding: duration, tempo, key, spectral balance, overall
    energy, and the normalised arrangement boundaries (the song's energy 'shape')."""
    structure = rec.get("structure") or {}
    dur = rec.get("duration") or 0.0
    secs = structure.get("sections") or []
    bounds = [round((s.get("start") or 0.0) / dur, 3) for s in secs if dur] if dur else []
    return {
        "durationSec": rec.get("duration"),
        "tempo": rec.get("bpm"),
        "key": _format_key(rec.get("key")),
        "spectral": spectral,
        "energy": energy,
        "sectionBounds": bounds,
    }


def to_audio_features(rec: dict) -> dict:
    """Map an analyze.analyze_file() record onto mixreflect's AudioFeatures shape
    (see src/lib/audio-analysis.ts). Returns the worker-side fields only;
    mixreflect tags `source: "worker"` itself."""
    efeat = rec.get("efeat") or {}
    structure = rec.get("structure") or {}
    mix_in = structure.get("mixIn") or {}
    vocal_env = structure.get("vocal") or []
    vocal_presence = (
        round(sum(vocal_env) / len(vocal_env), 3) if vocal_env else None
    )
    energy_1_10 = rec.get("energy")
    energy_norm = round(energy_1_10 / 10.0, 3) if energy_1_10 else None
    spectral = _derive_spectral(efeat)

    # Per-stem balance — only present when the demucs/Replicate stem pass ran (deep
    # reads). Lets the model say "the vocal sits under the guitars", not guess.
    stems_used = structure.get("structureSource") == "demucs"
    stem_mix = None
    if stems_used:
        drums_m = _mean(structure.get("drums"))
        bass_m = _mean(structure.get("bass"))
        voc_m = _mean(structure.get("vocal"))
        inst = [v for v in (drums_m, bass_m) if v is not None]
        stem_mix = {
            "drums": drums_m,
            "bass": bass_m,
            "vocals": voc_m,
            # >1 = vocal sits above the rhythm section; <1 = tucked under it.
            "vocalVsInstruments": (
                round(voc_m / (sum(inst) / len(inst)), 2)
                if voc_m is not None and inst and sum(inst) > 0 else None
            ),
        }

    feat = {
        "durationSec": rec.get("duration"),
        "tempo": rec.get("bpm"),
        "key": _format_key(rec.get("key")),
        # RMS dBFS — not a true integrated LUFS, but the right ballpark to feed the
        # model (the Spotify path feeds its `loudness` the same way).
        "loudnessLufs": efeat.get("rmsdb"),
        "energy": energy_norm,
        "spectral": spectral,
        "introLiftSec": mix_in.get("endSec"),
        "energyDips": _derive_dips(structure),
        # richer arrangement context (mixreflect's describeFeatures prints these):
        "sections": _sections_summary(structure),
        "vocalPresence": vocal_presence,
        "gridConfidence": structure.get("gridConfidence"),
        "stemsUsed": stems_used or None,
        "stemMix": stem_mix,
        # link-independent track identity (re-upload / version detection):
        "fingerprint": _acoustic_fingerprint(rec, spectral, energy_norm),
    }
    return {k: v for k, v in feat.items() if v is not None}


def _report_waveform(cache_dir: str, tid: str, cols: int = 1200) -> dict | None:
    """Compact 3-band waveform for the report page. analyze_file already computes
    the full-res rekordbox-style detail block (per-column LOW/MID/HIGH peaks +
    RMS body) and writes it to <cache>/wave/<id>.json — this max-pools it down to
    ~`cols` columns (a few KB of base64) so the report can draw a real frequency-
    split waveform. Max (not mean) pooling keeps the kicks/transients crisp."""
    try:
        import base64  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        with open(os.path.join(cache_dir, "wave", f"{tid}.json")) as f:
            detail = (json.load(f) or {}).get("detail") or {}
        n = int(detail.get("n") or 0)
        if not n or not all(detail.get(k) for k in ("lo", "mid", "hi", "amp")):
            return None

        def pool(b64s: str) -> str:
            a = np.frombuffer(base64.b64decode(b64s), dtype=np.uint8)
            if len(a) > cols:
                starts = np.linspace(0, len(a), cols, endpoint=False).astype(np.int64)
                a = np.maximum.reduceat(a, starts)
            return base64.b64encode(a.tobytes()).decode("ascii")

        return {
            "n": min(cols, n),
            "lo": pool(detail["lo"]),
            "mid": pool(detail["mid"]),
            "hi": pool(detail["hi"]),
            "amp": pool(detail["amp"]),
        }
    except Exception as e:  # noqa: BLE001 — waveform is decorative; never fail the read
        print(f"[worker] report waveform failed: {e}", flush=True)
        return None


def analyze_url(url: str, deep: bool = False) -> dict | None:
    """Full pipeline: download → transcode → analyze → map. None on any failure.

    `deep=True` runs the (slow) stem-separation pass — drums/bass/vocals/other
    structure + per-stem balance. mixreflect now requests deep for ALL score
    reads (instant included, ~$0.02/track on Replicate) — the paid gate is the
    deep prose, not the analysis. cache_dir = throwaway temp dir."""
    work_dir = tempfile.mkdtemp(prefix="djmix-worker-")
    try:
        acquired = acquire_wav(url, work_dir)
        if not acquired or not os.path.exists(acquired[0]):
            return None
        wav, src_dur = acquired
        rec = analyze.analyze_file(wav, work_dir, quick=not deep)
        feat = to_audio_features(rec)
        # When the analysis window (MAX_ANALYZE_SECS) truncated a longer track,
        # report the real source length so the read never claims an ending it
        # didn't hear and the UI can label the analysed span honestly.
        analyzed = feat.get("durationSec") or 0
        if src_dur and src_dur > analyzed + 5:
            feat["sourceDurationSec"] = round(src_dur, 1)
        wf = _report_waveform(work_dir, rec.get("id"))
        if wf:
            feat["waveform"] = wf
        return feat
    except Exception as e:  # noqa: BLE001
        print(f"[worker] analysis failed for {url}: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── HTTP ─────────────────────────────────────────────────────────────────────

def make_handler(token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            if not token:
                return True  # no secret configured → open (use only on a private network)
            auth = self.headers.get("Authorization", "")
            given = auth[7:] if auth.startswith("Bearer ") else ""
            return secrets.compare_digest(given, token)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                stems_on = False
                stem_backend = "none"
                try:
                    import stems  # noqa: PLC0415
                    stems_on = stems.available() and os.environ.get("DJMIX_STEMS", "1") != "0"
                    if stems_on:
                        stem_backend = "local-demucs" if getattr(stems, "HAVE_DEMUCS", False) else "replicate"
                except Exception:  # noqa: BLE001
                    pass
                self._json({"ok": True, "worker": "djmix", "stems": stems_on, "stemBackend": stem_backend, "ytCookies": _cookie_file() is not None, "proxy": bool(os.environ.get("YTDLP_PROXY")), "rev": "geo-retry-3"})
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/analyze":
                self._json({"error": "not found"}, 404)
                return
            if not self._authed():
                self._json({"error": "unauthorized"}, 401)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:  # noqa: BLE001
                self._json({"error": "invalid json"}, 400)
                return

            url = (payload.get("url") or "").strip()
            if not url:
                self._json({"error": "url required"}, 400)
                return
            deep = bool(payload.get("deep"))

            # One analysis at a time per box — wait briefly, then shed load so a
            # burst of uploads queues/degrades instead of OOM-crashing the worker.
            if not _ANALYZE_SEM.acquire(timeout=QUEUE_WAIT_SECS):
                print(f"[worker] busy — shedding {url}", flush=True)
                self._json({"features": None, "busy": True})
                return
            t0 = time.time()
            try:
                features = analyze_url(url, deep=deep)
            finally:
                _ANALYZE_SEM.release()
            took = round(time.time() - t0, 1)
            # Always reply 200 with a `{ features }` envelope: null means "couldn't
            # ground this one" so mixreflect falls back to its non-grounded read
            # instead of treating it as an outage.
            if features is None:
                print(f"[worker] no features for {url} ({took}s)", flush=True)
            else:
                print(f"[worker] analyzed {url} in {took}s", flush=True)
            self._json({"features": features, "took": took})

    return Handler


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description="djmix audio worker")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()

    token = os.environ.get("AUDIO_WORKER_SECRET")
    if not token:
        print("[worker] WARNING: AUDIO_WORKER_SECRET not set — /analyze is open", flush=True)
    if not shutil.which("ffmpeg"):
        print("[worker] WARNING: ffmpeg not found on PATH — analysis will fail", flush=True)
    print(f"[worker] youtube cookies: {'loaded ✓' if _cookie_file() else 'NOT set (YouTube links may be blocked)'}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(token))
    print(f"djmix worker — listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        server.shutdown()


if __name__ == "__main__":
    main()
