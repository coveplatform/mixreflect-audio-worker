"""djmix local analyzer.

Scans a folder of audio, reads tags + album art, and computes BPM, key
(Camelot), energy and an RGB (3-band) waveform per track. Results are cached
by path+mtime. Outputs <cache>/library.json (metadata), <cache>/wave/<id>.json
(waveform peaks) and <cache>/art/<id>.<ext> (extracted album art).

Usage: python analyze.py <folder> <cache_dir> [limit]
"""
import sys, os, json, zlib, base64, hashlib, traceback
import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile

import stems  # demucs stem-based structure (optional; falls back if not installed)

# Bump when the analysis output changes shape/meaning so cached rows (keyed by
# path+mtime+version) auto-recompute on the next scan. v2: real comb-filter
# beatgrid. v3: dynamics-preserving (RMS body, level-tied colour) hi-res waveform.
# v4: BPM octave-correction from grid lock strength.
# v5: BPM recovered by maximizing grid lock (fixes 2:3/half/double mis-reads).
# v6: fine-refine tagged BPMs too (rounded tags drift the blend off-beat).
# v7: precise period — barSec carried at 6 dp (was 3) so the phrase grid
#     extrapolates to the mix-out without ms-scale drift; rekordbox barSec from a
#     least-squares fit of the markers (exact, not rounded median tempo).
# v8: stem-based structure (demucs) — real drops/breakdowns/builds from
#     drums/bass/vocals/other instead of the mixed loudness; per-section vocal
#     salience; mix windows anchored to natural exits (last breakdown).
ANALYSIS_VERSION = 8

AUDIO_EXT = {'.mp3', '.flac', '.wav', '.aiff', '.aif', '.ogg', '.m4a'}
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def track_id(path: str) -> int:
    return zlib.crc32(path.encode('utf-8')) & 0x7FFFFFFF


_FP_CHUNK = 256 * 1024  # 256 KB head + tail — enough to disambiguate, cheap on huge files


def file_fingerprint(path: str) -> str:
    """Fast content hash that's stable across moves/renames and shareable between
    machines: sha1 over the file size + the first and last 256 KB. We deliberately
    avoid hashing the whole file (audio is large, hashing is the bottleneck) — head
    + tail + size collide only between genuinely identical-enough audio, which is
    exactly what we want a shared analysis cache to coalesce. Returns '' on error so
    callers fall back to the path+mtime key."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha1()
        h.update(str(size).encode('ascii'))
        with open(path, 'rb') as f:
            h.update(f.read(_FP_CHUNK))
            if size > _FP_CHUNK:
                f.seek(max(_FP_CHUNK, size - _FP_CHUNK))
                h.update(f.read(_FP_CHUNK))
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ''


def cache_keys(path: str, mtime=None):
    """Return (fp_key, fast_key) for the analysis cache.

    `fp_key` is the primary, content-addressed key (survives file moves/renames and
    is shareable across machines); `fast_key` is the legacy path+mtime key used as a
    cheap fast-path lookup. A caller should try `fast_key` first, then `fp_key`, and
    write the row under both so subsequent scans hit the cheap path. Old rows keyed
    by the legacy format simply miss and recompute."""
    if mtime is None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
    fast_key = f'{path}|{mtime}|v{ANALYSIS_VERSION}'
    fp = file_fingerprint(path)
    fp_key = f'{fp}|v{ANALYSIS_VERSION}' if fp else fast_key
    return fp_key, fast_key


def read_tags(path):
    title = artist = album = genre = key = ''
    bpm = None
    art = None
    art_ext = None
    try:
        mf = MutagenFile(path, easy=True)
        if mf is not None and mf.tags:
            g = lambda k: (mf.tags.get(k) or [''])[0]
            title, artist, album, genre = g('title'), g('artist'), g('album'), g('genre')
            bpmv = g('bpm')
            if bpmv:
                try:
                    bpm = float(bpmv)
                except ValueError:
                    pass
        # album art + key need the raw tags
        raw = MutagenFile(path)
        if raw is not None and raw.tags:
            for k, v in raw.tags.items():
                if k.startswith('APIC'):
                    art = v.data
                    art_ext = '.png' if 'png' in (v.mime or '').lower() else '.jpg'
                    break
            tkey = raw.tags.get('TKEY')
            if tkey:
                key = str(tkey.text[0]) if hasattr(tkey, 'text') else str(tkey)
            if raw.__class__.__name__ == 'FLAC' and raw.pictures:
                art = raw.pictures[0].data
                art_ext = '.png' if 'png' in raw.pictures[0].mime.lower() else '.jpg'
    except Exception:
        pass
    return title, artist, album, genre, bpm, key, art, art_ext


def estimate_bpm(x, sr):
    hop, win = 512, 1024
    n = 1 + (len(x) - win) // hop
    if n < 8:
        return None
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    frames = x[idx] * np.hanning(win).astype(np.float32)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    flux = flux - flux.mean()
    ac = np.correlate(flux, flux, mode='full')[len(flux) - 1:]
    fps = sr / hop
    lag_min = int(fps * 60 / 185)
    lag_max = int(fps * 60 / 70)
    seg = ac[lag_min:lag_max]
    if len(seg) == 0:
        return None
    best = lag_min + int(np.argmax(seg))
    bpm = 60 * fps / best
    while bpm < 90:
        bpm *= 2
    while bpm > 175:
        bpm /= 2
    return round(float(bpm), 1)


def estimate_key(x, sr):
    win, hop = 4096, 2048
    n = 1 + (len(x) - win) // hop
    if n < 4:
        return ''
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    frames = x[idx] * np.hanning(win).astype(np.float32)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    valid = freqs > 25
    pc = (np.round(69 + 12 * np.log2(freqs[valid] / 440.0)).astype(int)) % 12
    chroma = np.zeros(12)
    np.add.at(chroma, pc, mag[valid])
    if chroma.sum() <= 0:
        return ''
    chroma /= chroma.sum()
    best_score, best_pc, best_mode = -2, 0, 'maj'
    for shift in range(12):
        cs = np.roll(chroma, -shift)
        smaj = np.corrcoef(cs, KRUMHANSL_MAJ)[0, 1]
        smin = np.corrcoef(cs, KRUMHANSL_MIN)[0, 1]
        if smaj > best_score:
            best_score, best_pc, best_mode = smaj, shift, 'maj'
        if smin > best_score:
            best_score, best_pc, best_mode = smin, shift, 'min'
    return NOTE_NAMES[best_pc] + ('m' if best_mode == 'min' else '')


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def energy_features(x, amp, lo, mid, hi):
    """Raw, loudness-independent energy cues (per track). Loudness alone is a
    poor metric — modern masters are all brick-walled to the same RMS — so we
    measure what actually varies: spectral brightness, percussive movement, and
    sub-bass weight. These are normalized *across the library* later."""
    total = lo + mid + hi + 1e-9
    bright = float(np.mean(hi / total))            # high-band ratio ~ brightness
    sub = float(np.mean(lo / total))               # low-band ratio ~ floor weight
    flux = np.maximum(0, np.diff(amp))
    fluxn = float(np.mean(flux) / (np.mean(amp) + 1e-9))  # spectral movement / drive
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    rmsdb = float(20 * np.log10(rms))
    return {'bright': bright, 'sub': sub, 'flux': fluxn, 'rmsdb': rmsdb}


def energy_from_features(f):
    """Absolute fallback (used only when the library is too small to rank)."""
    b = _clamp01((f['bright'] - 0.04) / 0.22)
    fl = _clamp01(f['flux'] / 0.5)
    s = _clamp01((f['sub'] - 0.3) / 0.4)
    loud = _clamp01((f['rmsdb'] + 26) / 16)
    comp = 0.35 * b + 0.30 * fl + 0.15 * s + 0.20 * loud
    return int(max(1, min(10, round(1 + 9 * comp))))


def _rank01(vals):
    """Percentile rank 0..1 of each value within the list (ties keep order)."""
    a = np.asarray(vals, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a))
    return ranks / max(1, len(a) - 1)


def normalize_energy(records):
    """Library-relative energy: rank each cue across all tracks, blend, and map
    to 1..10 so the crate spreads instead of saturating. Loudness is weighted
    low (0.20) so brick-walled masters don't peg everything at 9."""
    idx = [i for i, r in enumerate(records) if r.get('efeat')]
    if len(idx) < 4:
        return  # too few to rank meaningfully — keep the per-track fallback
    feats = [records[i]['efeat'] for i in idx]
    bright = _rank01([f['bright'] for f in feats])
    flux = _rank01([f['flux'] for f in feats])
    sub = _rank01([f['sub'] for f in feats])
    loud = _rank01([f['rmsdb'] for f in feats])
    comp = 0.35 * bright + 0.30 * flux + 0.15 * sub + 0.20 * loud
    for k, i in enumerate(idx):
        records[i]['energy'] = int(min(10, max(1, round(1 + 9 * comp[k]))))


def _block_band_sums(x, win, hop, masks, block=1024):
    """Per-frame sums of |rfft(frame)| over the given boolean masks, computed in
    BLOCKS of frames. The all-at-once version materialized the whole framed
    track plus its complex128 spectrum (~300MB transient on a 480s window at
    the fine onset hop) and OOM-killed small boxes; block-wise it's O(block)
    (~20MB) with identical outputs to float32 precision."""
    n = max(1, 1 + (len(x) - win) // hop)
    w = np.hanning(win).astype(np.float32)
    outs = [np.empty(n, dtype=np.float32) for _ in masks]
    base = np.arange(win)
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        idx = base[None, :] + hop * np.arange(b0, b1)[:, None]
        np.clip(idx, 0, len(x) - 1, out=idx)
        frames = x[idx] * w[None, :]
        spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        del frames
        for m, out in zip(masks, outs):
            out[b0:b1] = spec[:, m].sum(axis=1)
        del spec
    return outs


def band_frames(x, sr, win=2048, hop=2048):
    """Per-frame energy in low/mid/high bands + total, computed once and reused
    by both the waveform and the structure analysis."""
    freqs = np.fft.rfftfreq(win, 1 / sr)
    amp, lo, mid, hi = _block_band_sums(
        x, win, hop,
        [np.ones(len(freqs), dtype=bool), freqs < 250,
         (freqs >= 250) & (freqs < 2500), freqs >= 2500],
    )
    fps = sr / hop
    return amp, lo, mid, hi, fps


def _band_color(Lp, Mp, Hp, body):
    """rekordbox-style colour: hue from the spectral *balance* (which bands
    dominate — lows→blue, mids→green/amber, highs→white), brightness from the
    actual level so quiet passages stay dim instead of every band self-
    normalizing to full. `Lp/Mp/Hp` are per-column band proportions (sum≈1),
    `body` the 0..1 loudness. Returns uint8 R,G,B (R=highs, G=mids, B=lows)."""
    bf = 0.32 + 0.68 * body  # brightness factor tied to level
    r = np.minimum(255, 255 * (0.12 + 0.88 * Hp) * bf)
    g = np.minimum(255, 238 * (0.14 + 0.86 * Mp) * bf)
    b = np.minimum(255, 235 * (0.18 + 0.82 * Lp) * bf)
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)


def rgb_waveform_from(amp, lo, mid, hi, bars=900):
    n = len(amp)
    groups = np.array_split(np.arange(n), min(bars, n))
    ab = np.array([amp[g].mean() for g in groups])
    lb = np.array([lo[g].mean() for g in groups])
    mb = np.array([mid[g].mean() for g in groups])
    hb = np.array([hi[g].mean() for g in groups])
    # body height from energy, normalized to a high percentile (not the max/p99)
    # and mildly gamma'd so dynamics survive — quiet stays short, drops stay tall.
    body = np.clip(ab / (np.percentile(ab, 96) + 1e-9), 0, 1) ** 0.9
    tot = lb + mb + hb + 1e-9
    Lp, Mp, Hp = lb / tot, mb / tot, hb / tot
    r, g, b = _band_color(Lp, Mp, Hp, body)
    out = []
    for i in range(len(ab)):
        out.append({'h': round(float(body[i]), 3), 'c': [int(r[i]), int(g[i]), int(b[i])]})
    return out


def detailed_waveform(x, sr, dur, rate=420, cap=240000):
    """High-resolution rekordbox-style 3-band waveform. For every column it stores
    the per-band PEAK of the LOW (<250 Hz), MID (250–2500 Hz) and HIGH (>2500 Hz)
    content. The renderer draws them centred & overlapping, biggest-behind — bass
    (blue) is the body, mids (amber) ride inside it, highs (white) form the bright
    core — so each kick reads as rekordbox's "spearhead" (white core, amber ring,
    blue body) rather than one flat-blue blob.

    Per-band PEAK (not RMS) keeps transients crisp; one shared gain (≈p99.5 of the
    broadband peak) keeps bass naturally dominant and the real dynamics intact.
    Bands are split by masking the full-track FFT. ~420 columns/sec so it stays
    sharp all the way down to 1-bar zoom. `amp` (RMS body) + `rgb` (blended) are
    kept for the audition lane."""
    L = len(x)
    n = int(min(cap, max(2000, dur * rate)))
    n = min(n, L)
    if n < 2:
        return None
    starts = np.linspace(0, L, n, endpoint=False).astype(np.int64)
    counts = np.diff(np.append(starts, L)).astype(np.float64)
    counts[counts <= 0] = 1.0

    ax = np.abs(x).astype(np.float32)
    bbpeak = np.maximum.reduceat(ax, starts)  # broadband peak → shared gain
    gain = float(np.percentile(bbpeak, 99.5)) + 1e-9
    del ax

    # Band-split via CHUNKED FFT masking → per-band time signal → per-column
    # PEAK. The old full-track version held the whole signal as complex128
    # (~3x the track in float64 temporaries per band) and OOM-killed small
    # boxes on long tracks; chunking keeps peak memory O(chunk) (~20MB) and
    # chunked FFTs are faster than one giant one. PAD samples of overlap are
    # computed at each edge and discarded, so filter edge effects never reach
    # a column. Output identical to the full-FFT version within float noise.
    CHUNK = 1 << 21  # ~2.1M samples per block
    PAD = 1 << 12

    # crossovers tuned to rekordbox's look: a tighter sub-bass band (less blue),
    # a wide mid band (more amber body), highs from 2 kHz (white transients).
    bands = [(0.0, 160.0), (160.0, 2000.0), (2000.0, sr)]
    outs = [np.zeros(n, dtype=np.float32) for _ in bands]

    pos = 0
    while pos < L:
        c0 = max(0, pos - PAD)
        win_hi = min(L, pos + CHUNK)
        c1 = min(L, win_hi + PAD)
        seg = np.ascontiguousarray(x[c0:c1], dtype=np.float32)
        S = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        # columns whose sample range lies fully inside [pos, win_hi)
        i0 = int(np.searchsorted(starts, pos))
        i1 = int(np.searchsorted(starts, win_hi))
        if i1 > i0:
            rel = starts[i0:i1] - c0
            for (f0, f1), out in zip(bands, outs):
                mask = (freqs >= f0) & (freqs < f1)
                xb = np.abs(np.fft.irfft(S * mask, n=len(seg)).astype(np.float32))
                # trim the tail pad so the last column's peak stops at win_hi
                m = np.maximum.reduceat(xb[: win_hi - c0], rel)
                np.maximum(out[i0:i1], m / gain, out=out[i0:i1])
                del xb
        del seg, S
        pos = win_hi
    lo_n, mid_n, hi_n = (np.clip(o, 0, 1) for o in outs)

    # RMS body + blended colour for the audition lane (single-envelope waveform).
    # float32 square (not float64): column sums are ~50 samples, well within
    # float32 precision, and it halves another full-track temporary.
    sq = x.astype(np.float32) ** 2
    rms = np.sqrt(np.add.reduceat(sq, starts) / counts)
    del sq
    amp_n = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0, 1) ** 0.85
    tot = lo_n + mid_n + hi_n + 1e-9
    r, g, b = _band_color(lo_n / tot, mid_n / tot, hi_n / tot, amp_n)
    rgb = np.stack([r, g, b], axis=1).reshape(-1)

    def b64(a):
        return base64.b64encode((a * 255).astype(np.uint8).tobytes()).decode('ascii')

    return {
        'n': int(n),
        'rate': rate,
        'lo': b64(lo_n),
        'mid': b64(mid_n),
        'hi': b64(hi_n),
        'amp': b64(amp_n),
        'rgb': base64.b64encode(rgb.tobytes()).decode('ascii'),
    }


ENV_BINS = 96  # resolution of the per-track structure envelopes


def _downsample(arr, bins):
    n = len(arr)
    if n == 0:
        return np.zeros(0)
    groups = np.array_split(np.arange(n), min(bins, n))
    return np.array([arr[g].mean() for g in groups])


def _norm(arr, pct=99):
    if len(arr) == 0:
        return arr
    return np.clip(arr / (np.percentile(arr, pct) + 1e-9), 0, 1)


def estimate_beat_offset(amp, fps, bpm):
    """Seconds to the strongest early onset (a coarse downbeat-phase guess).
    Fallback only — `analyze_beatgrid` supersedes this when it runs."""
    if not bpm or len(amp) < 4:
        return 0.0
    flux = np.maximum(0, np.diff(amp))
    look = max(4, int(fps * (60.0 / bpm) * 8))  # first ~8 beats
    seg = flux[:look]
    if len(seg) == 0:
        return 0.0
    return round(float(np.argmax(seg)) / fps, 3)


def onset_envelopes(x, sr, hop=512, win=1024):
    """Broadband + low-band onset (spectral-flux) envelopes at a fine hop (~12ms
    @ 44.1k). The low band isolates kicks, which land on the downbeat — used to
    pick beat 1 of the bar. Returns (full_flux, low_flux, fps) or None."""
    n = 1 + (len(x) - win) // hop
    if n < 16:
        return None
    # Block-wise (see _block_band_sums): this fine hop frames the whole track —
    # the all-at-once spectrum was the single biggest allocation in the analyzer.
    freqs = np.fft.rfftfreq(win, 1 / sr)
    full, low = _block_band_sums(
        x, win, hop, [np.ones(len(freqs), dtype=bool), freqs < 150]
    )
    fflux = np.concatenate([[0.0], np.maximum(0, np.diff(full))])
    lflux = np.concatenate([[0.0], np.maximum(0, np.diff(low))])
    return fflux, lflux, sr / hop


def _circ_smooth(a, k=5):
    """Circular moving-average smooth (the phase histogram wraps at the beat)."""
    if len(a) < k:
        return a
    pad = np.concatenate([a[-k:], a, a[:k]])
    sm = np.convolve(pad, np.ones(k) / k, mode='same')
    return sm[k:-k]


def _parabolic_offset(a, i):
    """Sub-bin peak offset in (-0.5, 0.5) via parabolic interpolation (circular)."""
    n = len(a)
    l, c, r = a[(i - 1) % n], a[i], a[(i + 1) % n]
    denom = l - 2 * c + r
    if denom == 0:
        return 0.0
    return float(np.clip(0.5 * (l - r) / denom, -0.5, 0.5))


def _grid_strength(fflux, fps, bpm, K=192):
    """How hard the onsets lock to a grid at this BPM: peak/mean of the onset
    envelope folded over one beat period (a comb filter). Razor-sharp — the right
    tempo scores ~5-15x, a wrong one ~1.2 — so it's a great objective to optimize
    BPM against. bincount keeps the fold C-fast for a fine tempo scan."""
    period = (60.0 / bpm) * fps
    if period < 2 or len(fflux) < 8:
        return 0.0
    bins = np.minimum(K - 1, ((np.arange(len(fflux)) / period) % 1.0 * K).astype(np.int64))
    acc = _circ_smooth(np.bincount(bins, weights=fflux, minlength=K)[:K], 5)
    m = float(acc.mean())
    return float(acc.max() / (m + 1e-9)) if m > 0 else 0.0


def _tempo_pref(bpm):
    """Gentle preference for the common dance range — only breaks half/double
    *ties*; a strong grid lock always wins. Keeps a true 87↔174 ambiguity sane."""
    if 110 <= bpm <= 162:
        return 1.0
    if bpm < 110:
        return 0.82 + 0.18 * (bpm - 70) / 40
    return 0.82 + 0.18 * (185 - bpm) / 23


def refine_bpm(fflux, fps, lo=70.0, hi=185.0):
    """Find the BPM whose grid the onsets lock onto hardest — far more accurate
    than autocorrelation, which routinely picks a 2:3/half/double of the real
    tempo. Coarse 0.1-BPM scan (the lock peak is <0.5 BPM wide) then a fine local
    refine. Returns (bpm, strength)."""
    coarse = np.arange(lo, hi, 0.1)
    sc = np.array([_grid_strength(fflux, fps, b) * _tempo_pref(b) for b in coarse])
    if not len(sc):
        return None
    b0 = float(coarse[int(np.argmax(sc))])
    fine = np.arange(max(lo, b0 - 0.3), min(hi, b0 + 0.3), 0.02)
    sf = np.array([_grid_strength(fflux, fps, b) * _tempo_pref(b) for b in fine])
    bpm = float(fine[int(np.argmax(sf))]) if len(sf) else b0
    return round(bpm, 2), _grid_strength(fflux, fps, bpm)


def _grid_at(fflux, lflux, fps, bpm, total_sec):
    """Fit the beatgrid at a *given* BPM: comb-filter phase, then pick which of
    the 4 beats is the downbeat from where the kicks (low-band onsets) sit."""
    nb = len(fflux)
    beat_sec = 60.0 / bpm
    period = beat_sec * fps  # frames per beat
    if period < 2:
        return None
    K = 192
    bins = np.minimum(K - 1, ((np.arange(nb) / period) % 1.0 * K).astype(np.int64))
    acc = _circ_smooth(np.bincount(bins, weights=fflux, minlength=K)[:K], 5)
    best = int(np.argmax(acc))
    beat_frac = ((best + _parabolic_offset(acc, best)) / K) % 1.0
    beat_offset = beat_frac * beat_sec
    strength = float(acc.max() / (acc.mean() + 1e-9))

    bar_sec = beat_sec * 4
    best_j, best_e = 0, -1.0
    for j in range(4):
        off = beat_offset + j * beat_sec
        ts = np.arange(off, total_sec, bar_sec)
        fi = np.clip((ts * fps).astype(int), 0, nb - 1)
        e = float(lflux[fi].mean()) if len(fi) else 0.0
        if e > best_e:
            best_e, best_j = e, j
    downbeat_offset = (beat_offset + best_j * beat_sec) % bar_sec
    return {
        'beatOffset': round(beat_offset, 3),
        'downbeatOffset': round(downbeat_offset, 3),
        'gridStrength': round(strength, 2),
    }


def analyze_beatgrid(x, sr, bpm, allow_octave=True):
    """A real beatgrid. For untagged tracks (`allow_octave`) the BPM is recovered
    by maximizing grid lock (`refine_bpm`) rather than trusting autocorrelation,
    which routinely lands on a 2:3/half/double of the true tempo (e.g. reads 89
    for a 134 track). A tagged BPM's value is trusted; we just fit the phase.
    Returns beatOffset, downbeatOffset, gridStrength, gridConfidence, bpm."""
    if not bpm:
        return None
    env = onset_envelopes(x, sr)
    if env is None:
        return None
    fflux, lflux, fps = env
    total = len(x) / sr
    if allow_octave:
        refined = refine_bpm(fflux, fps)  # untagged: full grid-locked search
        if refined:
            bpm = refined[0]
    else:
        # Tagged: tags are often rounded (137.0 for a track that's really 137.7),
        # and a 0.7-BPM error drifts ~140ms over a 16-bar blend. Fine-refine within
        # ±1.6 BPM of the tag to lock the exact tempo without leaving its octave.
        refined = refine_bpm(fflux, fps, lo=max(60.0, bpm - 1.6), hi=min(210.0, bpm + 1.6))
        if refined and refined[1] > _grid_strength(fflux, fps, bpm) * 1.04:
            bpm = refined[0]
    g = _grid_at(fflux, lflux, fps, bpm, total)
    if g is None:
        return None
    out = dict(g)
    out['bpm'] = round(float(bpm), 2)
    out['gridConfidence'] = 'high' if g['gridStrength'] > 3.0 else 'low'
    return out


def _merge_sections(labels, sec_per_bin, energy, min_sec=12.0):
    """Collapse a per-bin label array into contiguous sections, then absorb
    fragments shorter than `min_sec` into a neighbour so real tracks yield a
    handful of meaningful sections, not dozens of build/body flickers."""
    runs = []  # [start_bin, end_bin, label]
    if len(labels) == 0:
        return []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append([start, i, labels[start]])
            start = i
    # absorb short interior runs into the longer adjacent neighbour
    min_bins = max(1, int(min_sec / sec_per_bin)) if sec_per_bin else 1
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, r in enumerate(runs):
            if r[2] in ('intro', 'outro'):
                continue
            if r[1] - r[0] >= min_bins:
                continue
            prev = runs[j - 1] if j > 0 else None
            nxt = runs[j + 1] if j < len(runs) - 1 else None
            host = prev if (prev and (not nxt or (prev[1] - prev[0]) >= (nxt[1] - nxt[0]))) else nxt
            if not host:
                continue
            host[0] = min(host[0], r[0])
            host[1] = max(host[1], r[1])
            runs.pop(j)
            changed = True
            break
    # coalesce adjacent runs that ended up with the same label
    coalesced = []
    for r in runs:
        if coalesced and coalesced[-1][2] == r[2]:
            coalesced[-1][1] = r[1]
        else:
            coalesced.append(r)
    runs = coalesced
    out = []
    for s, e, lab in runs:
        seg = energy[s:e]
        out.append({
            'start': round(s * sec_per_bin, 1),
            'end': round(e * sec_per_bin, 1),
            'kind': lab,
            'energy': round(float(seg.mean()) if len(seg) else 0.0, 3),
        })
    return out


def analyze_structure(amp, lo, mid, hi, fps, bpm, dur, beatgrid=None):
    """The §6b mixability data: sections, mix windows, and bass/energy/vocal
    envelopes — derived from the band energies we already decoded. `beatgrid`
    (from analyze_beatgrid) supplies the real grid phase + downbeat when present."""
    bar_sec = (60.0 / bpm) * 4 if bpm else None
    energy = _downsample(_norm(amp), ENV_BINS)
    bass = _downsample(_norm(lo), ENV_BINS)
    vocal = _downsample(_norm(mid / (amp + 1e-9)), ENV_BINS)  # mid salience ~ weak vocal proxy
    nb = len(energy)
    if nb == 0:
        return None
    sec_per_bin = dur / nb if nb else dur

    # smooth both envelopes before sectioning
    def _smooth(a):
        return np.convolve(a, np.ones(5) / 5, mode='same') if nb >= 5 else a

    es = _smooth(energy)
    bs = _smooth(bass)
    # robust peak (95th pct, not a single spike) so loud tracks don't all read "peak"
    peak = float(np.percentile(es, 95)) or 1.0

    # intro/outro = the lead-in / lead-out where the track hasn't fully kicked in
    on_thr = 0.45 * peak
    first_on = next((i for i in range(nb) if es[i] >= on_thr), 0)
    last_on = next((i for i in range(nb - 1, -1, -1) if es[i] >= on_thr), nb - 1)

    # bass reference from the *active* middle (typical "full bass" level)
    act = bs[first_on:last_on + 1]
    bass_ref = float(np.percentile(act, 75)) if len(act) else float(np.percentile(bs, 75))
    bass_out = 0.45 * (bass_ref or 1.0)  # below this, the low end has dropped out
    drop_e = 0.55 * peak                 # full-energy threshold

    def window(a_bin, b_bin):
        start = round(a_bin * sec_per_bin, 1)
        end = round(b_bin * sec_per_bin, 1)
        bars = int(round((end - start) / bar_sec)) if bar_sec else None
        return {'startSec': start, 'endSec': end, 'bars': bars}

    mix_in = window(0, first_on)
    mix_out = window(last_on, nb)

    # per-bin labels: bass-aware. A "drop" needs both energy AND the low end;
    # a "breakdown" is where the bass falls away mid-track; "build" is a rise.
    labels = []
    for i in range(nb):
        if i < first_on:
            lab = 'intro'
        elif i > last_on:
            lab = 'outro'
        elif bs[i] < bass_out and es[i] < 0.8 * peak:
            lab = 'breakdown'           # low end gone — bass-stripped section
        elif es[i] >= drop_e and bs[i] >= bass_out:
            lab = 'drop'                # full: energy + bass present
        elif i > 0 and es[i] > es[i - 1] + 0.01:
            lab = 'build'               # rising into the next drop
        else:
            lab = 'drop' if es[i] >= 0.5 * peak else 'build'
        labels.append(lab)
    sections = _merge_sections(labels, sec_per_bin, es)

    out = {
        'dur': round(float(dur), 1),
        # 6 dp: the engine extrapolates the phrase grid as downbeat + k·barSec to
        # the mix-out (~24 phrases), so coarse rounding here drifts the blend off
        # the beat by the outro. rekordbox's precise barSec overrides via update().
        'barSec': round(bar_sec, 6) if bar_sec else None,
        'beatOffset': estimate_beat_offset(amp, fps, bpm),
        'sections': sections,
        'mixIn': mix_in,
        'mixOut': mix_out,
        'bass': [round(float(v), 3) for v in bass],
        'energyEnv': [round(float(v), 3) for v in energy],
        'vocal': [round(float(v), 3) for v in vocal],
        'confidence': 'full' if bpm else 'lite',
    }
    if beatgrid:
        # real comb-filter grid supersedes the single-onset beatOffset fallback
        out.update(beatgrid)
    return out


def stems_enabled():
    """True when the slow demucs stem pass should run (installed + not disabled)."""
    return stems.available() and os.environ.get('DJMIX_STEMS', '1') != '0'


def _apply_stem_structure(path, bpm, structure, dur):
    """Upgrade `structure` in place with stem-based sectioning (drums/bass/vocals/
    other). The grid (barSec/downbeat) and key/bpm stay as computed; only the
    sections, mix windows and envelopes are replaced. Returns True if upgraded."""
    if not (structure and structure.get('barSec')):
        return False
    try:
        down = structure.get('downbeatOffset') or structure.get('beatOffset') or 0.0
        sep = stems.separate(path)
        if sep:
            st = stems.structure_from_stems(sep[0], sep[1], bpm, structure['barSec'], down, dur)
            if st and st.get('sections'):
                for k in ('sections', 'mixIn', 'mixOut', 'bass', 'energyEnv', 'vocal', 'drums', 'structureSource'):
                    if k in st:
                        structure[k] = st[k]
                print(f'  stems: {len(st["sections"])} sections', flush=True)
                return True
    except Exception as e:  # noqa: BLE001
        print(f'  stem structure skipped: {e}', flush=True)
    return False


def deepen_structure(rec, path):
    """Second (slow) phase of progressive analysis: run the demucs stem pass on an
    already-quick-analyzed record and replace its loudness sectioning with the
    stem-based structure. Returns the (mutated) record, or the unchanged record if
    stems are unavailable/disabled or separation fails. Safe to call repeatedly."""
    if not stems_enabled():
        return rec
    structure = rec.get('structure')
    if not structure or rec.get('structure', {}).get('structureSource') == 'demucs':
        return rec  # nothing to deepen / already deep
    if _apply_stem_structure(path, rec.get('bpm'), structure, rec.get('duration') or 0.0):
        rec['structure'] = structure
    return rec


def analyze_file(path, cache_dir, rb_grid=None, quick=False):
    """Analyze one track. `rb_grid` (from rekordbox.py) supplies rekordbox's exact
    BPM + beatgrid (downbeat) — when present we trust it over our own estimate,
    which is what makes auditions lock; we still compute waveform/energy/structure
    locally.

    `quick=True` skips the slow demucs stem pass (BPM/key/grid/energy/waveform +
    loudness-based structure only) so the agent can publish grids + waveforms ASAP;
    a later `deepen_structure(rec, path)` upgrades the sectioning in the background."""
    tid = track_id(path)
    title, artist, album, genre, tag_bpm, tag_key, art, art_ext = read_tags(path)
    if not title:
        title = os.path.splitext(os.path.basename(path))[0]

    data, sr = sf.read(path, always_2d=True, dtype='float32')
    x = data.mean(axis=1)
    dur = round(len(x) / sr, 1)
    # analyse a central window (speed); waveform uses the whole track
    win_s = 120
    if len(x) > win_s * sr:
        start = (len(x) - win_s * sr) // 2
        xa = x[start:start + win_s * sr]
    else:
        xa = x

    amp, lo, mid, hi, fps = band_frames(x, sr)
    key = tag_key or estimate_key(xa, sr)
    efeat = energy_features(x, amp, lo, mid, hi)
    energy = energy_from_features(efeat)  # provisional; normalize_energy refines across the library
    wave = rgb_waveform_from(amp, lo, mid, hi)
    detail = detailed_waveform(x, sr, dur)
    if rb_grid:
        # rekordbox is the source of truth: exact BPM + marked downbeat. barSec is
        # the least-squares period (carried separately from the 2-dp display bpm so
        # the blend stays locked all the way to the mix-out).
        bpm = rb_grid['bpm']
        beatgrid = {
            'beatOffset': rb_grid['beatOffset'],
            'downbeatOffset': rb_grid['downbeatOffset'],
            'gridConfidence': 'high',
            'gridSource': 'rekordbox',
        }
        if rb_grid.get('barSec'):
            beatgrid['barSec'] = rb_grid['barSec']  # overrides the bpm-derived barSec
        if rb_grid.get('variableGrid'):
            beatgrid['variableGrid'] = True
        if rb_grid.get('downbeats'):
            beatgrid['downbeats'] = rb_grid['downbeats']  # real kicks → audition anchors to them
    else:
        bpm = tag_bpm or estimate_bpm(xa, sr)
        # trust a tagged BPM's octave; only octave-correct our own estimate
        beatgrid = analyze_beatgrid(x, sr, bpm, allow_octave=(tag_bpm is None))
        if beatgrid:
            bpm = beatgrid.pop('bpm', bpm)  # adopt the octave that locks best
    structure = analyze_structure(amp, lo, mid, hi, fps, bpm, dur, beatgrid)

    # When demucs is installed, replace the loudness-based sectioning with
    # stem-based structure (drums/bass/vocals/other) — it reads the arrangement the
    # way a DJ does, so flat-energy tracks still segment into real drops/breakdowns.
    # The grid (barSec/downbeat) and key/bpm stay as computed; only the sections,
    # mix windows and envelopes are upgraded. Skipped in `quick` mode (the agent
    # runs it later via deepen_structure so grids/waveforms publish immediately).
    if not quick and stems_enabled():
        _apply_stem_structure(path, bpm, structure, dur)

    art_url = None
    if art:
        os.makedirs(os.path.join(cache_dir, 'art'), exist_ok=True)
        ap = os.path.join(cache_dir, 'art', f'{tid}{art_ext or ".jpg"}')
        with open(ap, 'wb') as f:
            f.write(art)
        art_url = os.path.basename(ap)

    os.makedirs(os.path.join(cache_dir, 'wave'), exist_ok=True)
    with open(os.path.join(cache_dir, 'wave', f'{tid}.json'), 'w') as f:
        json.dump({'bars': wave, 'detail': detail, 'dur': dur}, f)

    return {
        'id': tid, 'title': title, 'artist': artist, 'album': album, 'genre': genre,
        'bpm': bpm, 'key': key, 'energy': energy, 'duration': dur,
        'location': path, 'art': art_url, 'structure': structure, 'efeat': efeat,
    }


def main():
    folder, cache_dir = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'cache.json')
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}

    files = []
    for root, _dirs, names in os.walk(folder):
        if 'System Volume Information' in root:
            continue
        for nm in names:
            if nm.startswith('._') or nm.startswith('.'):
                continue
            if os.path.splitext(nm)[1].lower() in AUDIO_EXT:
                files.append(os.path.join(root, nm))
    files.sort()
    if limit:
        files = files[:limit]

    records, done, errors = [], 0, 0
    for path in files:
        try:
            mt = os.path.getmtime(path)
            fp_key, fast_key = cache_keys(path, mt)
            # fast-path: path+mtime hit avoids re-fingerprinting unchanged files
            if fast_key in cache:
                rec = cache[fast_key]
            elif fp_key in cache:  # content hit — file moved/renamed or shared cache
                rec = cache[fp_key]
            else:
                rec = analyze_file(path, cache_dir)
            # write under both keys so future scans hit either path
            cache[fp_key] = rec
            cache[fast_key] = rec
            records.append(rec)
            done += 1
            print(f'[{done}/{len(files)}] {os.path.basename(path)}', flush=True)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f'ERR {os.path.basename(path)}: {e}', flush=True)
            traceback.print_exc()

    normalize_energy(records)  # library-relative energy across the whole scan
    json.dump(cache, open(cache_path, 'w'))
    json.dump(records, open(os.path.join(cache_dir, 'library.json'), 'w'))
    print(f'DONE {len(records)} tracks, {errors} errors -> {cache_dir}/library.json', flush=True)


if __name__ == '__main__':
    main()
