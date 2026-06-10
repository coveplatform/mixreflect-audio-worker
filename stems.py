"""Stem-based structure analysis.

Uses **demucs** (an audio ML model) to split a track into drums / bass / vocals /
other, then reads the arrangement the way a DJ hears it — where the drums and bass
drop out (breakdown), where it's full (drop), where the vocal sits, where it
builds — instead of guessing from the mixed loudness (which collapses on tracks
that are loud start-to-finish).

Heavy (torch). Optional: analyze.py falls back to the energy-based detector when
demucs isn't installed or separation fails.
"""
import numpy as np
import soundfile as sf

try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    HAVE_DEMUCS = True
except Exception:  # noqa: BLE001
    HAVE_DEMUCS = False

_MODEL = None
_DEVICE = None
ENV_BINS = 96  # match analyze.ENV_BINS so the engine's envelopes line up

import os as _os
# htdemucs = best/slowest; mdx_q = ~2x faster (quantized). Structure only needs
# rough per-stem energy, so the faster model is usually fine. Override via env.
MODEL_NAME = _os.environ.get('DJMIX_DEMUCS_MODEL', 'htdemucs')

# Replicate hosted Demucs — lets a CPU-only box (e.g. Render) get GPU stem
# separation without torch/demucs installed locally. Set REPLICATE_API_TOKEN to
# enable; pin the model+version with REPLICATE_DEMUCS_MODEL. structure_from_stems
# (pure numpy) still runs locally on the returned stems.
REPLICATE_DEMUCS_MODEL = _os.environ.get(
    'REPLICATE_DEMUCS_MODEL',
    'cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953',
)
# Hard ceiling on the Replicate round-trip (queue + run). Cold starts were
# observed at 6+ minutes — past the app's whole serverless budget — and an
# unbounded wait pinned the worker while the caller had long since given up.
# On timeout the prediction is cancelled (stop paying) and the read keeps its
# loudness structure: stems are an upgrade, never a dependency.
REPLICATE_TIMEOUT_SECS = int(_os.environ.get('REPLICATE_TIMEOUT_SECS', '150'))


def _replicate_enabled():
    return bool(_os.environ.get('REPLICATE_API_TOKEN'))


def available():
    """Stems can run if local demucs is installed OR a Replicate token is set."""
    return HAVE_DEMUCS or _replicate_enabled()


def _separate_replicate(path):
    """Separate via Replicate's hosted Demucs. Returns ({drums,bass,vocals,other:
    float32}, sr) or None. No torch needed — the GPU work runs on Replicate; we
    just download the stems and read them with soundfile."""
    try:
        import replicate  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        import time  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f'  replicate import failed: {e}', flush=True)
        return None
    try:
        # predictions.create + bounded poll instead of replicate.run(): run()
        # blocks until the prediction finishes, however long the GPU queue is.
        version = REPLICATE_DEMUCS_MODEL.split(':', 1)[-1]
        with open(path, 'rb') as fh:
            pred = replicate.predictions.create(version=version, input={'audio': fh})
        deadline = time.time() + REPLICATE_TIMEOUT_SECS
        while pred.status not in ('succeeded', 'failed', 'canceled'):
            if time.time() > deadline:
                try:
                    pred.cancel()
                except Exception:  # noqa: BLE001
                    pass
                print(f'  replicate demucs timed out ({REPLICATE_TIMEOUT_SECS}s) — skipping stems', flush=True)
                return None
            time.sleep(3)
            pred.reload()
        if pred.status != 'succeeded':
            print(f'  replicate demucs {pred.status}: {pred.error}', flush=True)
            return None
        out = pred.output
    except Exception as e:  # noqa: BLE001
        print(f'  replicate demucs failed: {e}', flush=True)
        return None

    # Output is a map of stem-name -> URL (FileOutput in newer clients). Normalise
    # to {name: url}. Accept a few key spellings demucs variants use.
    def _url(v):
        return v.url if hasattr(v, 'url') else (v if isinstance(v, str) else None)

    urls = {}
    if isinstance(out, dict):
        for name in ('drums', 'bass', 'vocals', 'other'):
            for k in (name, f'{name}.mp3', f'{name}.wav'):
                if k in out and _url(out[k]):
                    urls[name] = _url(out[k]); break
    if not urls:
        print(f'  replicate demucs: unexpected output shape {type(out)}', flush=True)
        return None

    tmp = tempfile.mkdtemp(prefix='djmix-stems-')
    sigs = {}
    sr_common = None
    try:
        for name, url in urls.items():
            dst = _os.path.join(tmp, f'{name}.wav')
            urllib.request.urlretrieve(url, dst)
            # Memory discipline — this runs on a small box (Render starter =
            # 512MB) and full-length 44.1k stereo float32 stems are ~74MB each;
            # loading them naively OOM-killed the worker (no traceback, just a
            # restart). Mono immediately, halve the rate (structure envelopes
            # don't need 44.1k), and free every intermediate before the next
            # stem downloads.
            data, sr = sf.read(dst, always_2d=True, dtype='float32')
            mono = data.mean(axis=1, dtype=np.float32)
            del data
            dec = 2 if sr >= 44100 else 1
            sigs[name] = np.ascontiguousarray(mono[::dec])
            del mono
            try:
                _os.remove(dst)
            except OSError:
                pass
            sr_common = sr_common or sr // dec
        # demucs stems share a sample rate; align lengths defensively.
        if sigs:
            n = min(len(s) for s in sigs.values())
            sigs = {k: v[:n] for k, v in sigs.items()}
            return sigs, int(sr_common)
    except Exception as e:  # noqa: BLE001
        print(f'  replicate stem download failed: {e}', flush=True)
    finally:
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp, ignore_errors=True)
    return None


def _pick_device():
    """Pick the fastest available torch device for separation: CUDA (NVIDIA) >
    Apple MPS > CPU. `DJMIX_DEMUCS_DEVICE` overrides (e.g. 'cuda'/'mps'/'cpu').
    Any probe that throws falls back to CPU so a broken GPU/driver never kills the
    stem pass. Memoized — picked (and logged) once."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    dev = 'cpu'
    override = _os.environ.get('DJMIX_DEMUCS_DEVICE', '').strip().lower()
    if override:
        dev = override
    else:
        try:
            if torch.cuda.is_available():
                dev = 'cuda'
            elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                dev = 'mps'
        except Exception:  # noqa: BLE001 — any probe failure → CPU
            dev = 'cpu'
    _DEVICE = dev
    print(f'[stems] device={dev}', flush=True)
    return _DEVICE


def _model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        m = get_model(MODEL_NAME)
        dev = _pick_device()
        try:
            m.to(dev).eval()
        except Exception as e:  # noqa: BLE001 — GPU OOM/init failure → CPU fallback
            print(f'[stems] device={dev} init failed ({e}); falling back to cpu', flush=True)
            dev = _DEVICE = 'cpu'
            m.cpu().eval()
        _MODEL = m
    return _MODEL


def separate(path):
    """Split into mono stems. Returns ({drums,bass,vocals,other: float32}, sr) or None.

    Prefers a local demucs install (no network); falls back to Replicate's hosted
    GPU Demucs when REPLICATE_API_TOKEN is set and torch isn't installed."""
    if not HAVE_DEMUCS:
        if _replicate_enabled():
            return _separate_replicate(path)
        return None
    try:
        m = _model()
        dev = _pick_device()
        sr_m = int(m.samplerate)
        # decode with soundfile (torchaudio.load needs torchcodec in 2.11), then
        # hand the tensor straight to demucs. [samples, channels] → [channels, samples]
        data, sr = sf.read(path, always_2d=True, dtype='float32')
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        if sr != sr_m:
            wav = torchaudio.functional.resample(wav, sr, sr_m)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)  # demucs expects stereo
        elif wav.shape[0] > 2:
            wav = wav[:2]
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8
        wn = (wav - mean) / std
        with torch.no_grad():
            src = apply_model(m, wn[None], shifts=0, split=True, overlap=0.1, device=dev, progress=False)[0]
        src = src * std + mean
        out = {}
        for i, name in enumerate(m.sources):  # ['drums','bass','other','vocals']
            out[name] = src[i].mean(0).cpu().numpy().astype(np.float32)
        return out, sr_m
    except Exception as e:  # noqa: BLE001
        print(f'  demucs failed: {e}', flush=True)
        return None


def _beat_rms(sig, sr, beat_times):
    """RMS of `sig` in each [beat, next-beat) interval."""
    n = len(beat_times)
    out = np.zeros(n, dtype=np.float64)
    L = len(sig)
    for i in range(n):
        a = int(beat_times[i] * sr)
        nxt = beat_times[i + 1] if i + 1 < n else beat_times[i] + (beat_times[i] - beat_times[i - 1] if i > 0 else 0.5)
        b = min(int(nxt * sr), L)
        if b > a >= 0:
            seg = sig[a:b]
            out[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return out


def _norm(a):
    p = np.percentile(a, 90) + 1e-9
    return np.clip(a / p, 0, 1.5)


def _resample(arr, bins):
    if len(arr) == 0:
        return [0.0] * bins
    idx = np.linspace(0, len(arr), bins, endpoint=False).astype(int)
    idx = np.clip(idx, 0, len(arr) - 1)
    return [round(float(arr[i]), 3) for i in idx]


def structure_from_stems(stems, sr, bpm, bar_sec, downbeat, dur):
    """Build the §6b structure dict from separated stems. Same shape as
    analyze_structure so it drops straight into analyze_file."""
    if not bar_sec or bar_sec <= 0:
        return None
    beat_sec = bar_sec / 4
    drums = stems.get('drums')
    bass = stems.get('bass')
    vocals = stems.get('vocals')
    other = stems.get('other')
    if drums is None or bass is None:
        return None

    # beat grid (phase from the real downbeat), one frame per beat
    phase = downbeat % beat_sec
    beats = np.arange(phase, dur, beat_sec)
    if len(beats) < 8:
        return None
    drm = _norm(_beat_rms(drums, sr, beats))
    bss = _norm(_beat_rms(bass, sr, beats))
    voc = _norm(_beat_rms(vocals, sr, beats)) if vocals is not None else np.zeros(len(beats))
    oth = _norm(_beat_rms(other, sr, beats)) if other is not None else np.zeros(len(beats))
    total = _norm(drm + bss + voc + oth)
    nb = len(beats)

    def sm(a, w=4):
        k = np.ones(w) / w
        return np.convolve(a, k, mode='same')

    drmS, bssS, vocS, othS, totS = sm(drm), sm(bss), voc, sm(oth), sm(total)

    # presence flags (relative to the track's own typical levels)
    DRUM_ON, BASS_ON, VOC_ON = 0.33, 0.30, 0.40

    # --- boundaries: novelty on the stem feature vectors, snapped to 4-bar phrases ---
    feat = np.stack([drmS, bssS, vocS, othS], axis=1)
    W = 16  # 4 bars
    novelty = np.zeros(nb)
    for i in range(nb):
        a0, a1 = max(0, i - W), i
        b0, b1 = i, min(nb, i + W)
        if a1 > a0 and b1 > b0:
            novelty[i] = float(np.linalg.norm(feat[a0:a1].mean(0) - feat[b0:b1].mean(0)))
    bound = {0, nb}
    thr = float(np.percentile(novelty, 80)) + 1e-6
    order = np.argsort(novelty)[::-1]
    for i in order:
        if novelty[i] < thr:
            break
        if all(abs(i - bb) >= 16 for bb in bound):  # ≥4 bars apart
            bound.add(int(i))
    # snap every boundary to the nearest 4-bar line
    bars4 = 16
    bound = sorted({0, nb} | {int(round(b / bars4) * bars4) for b in bound if 0 < b < nb})
    bound = [b for b in bound if 0 <= b <= nb]
    if bound[0] != 0:
        bound = [0] + bound
    if bound[-1] != nb:
        bound = bound + [nb]

    # --- classify each segment by what's actually playing ---
    def classify(s, e):
        d, b, v, t = drmS[s:e].mean(), bssS[s:e].mean(), vocS[s:e].mean(), totS[s:e].mean()
        rising = totS[e - 1] - totS[s] > 0.15 if e - 1 > s else False
        drums_on, bass_on = d > DRUM_ON, b > BASS_ON
        if s == 0 and (not drums_on or t < 0.35):
            kind = 'intro'
        elif e >= nb and t < 0.4:
            kind = 'outro'
        elif not drums_on and not bass_on:
            kind = 'breakdown'
        elif not bass_on and t < 0.7:
            kind = 'breakdown'  # bass-stripped section
        elif rising and not (drums_on and bass_on and t > 0.6):
            kind = 'build'
        elif drums_on and bass_on and t > 0.5:
            kind = 'drop'
        else:
            kind = 'drop' if t > 0.45 else 'build'
        return kind, float(t), float(v)

    sections = []
    for s, e in zip(bound[:-1], bound[1:]):
        if e <= s:
            continue
        kind, energy, vmean = classify(s, e)
        sections.append({
            'start': round(float(beats[s]) if s < nb else float(dur), 1),
            'end': round(float(beats[e]) if e < nb else float(dur), 1),
            'kind': kind,
            'energy': round(energy, 3),
            'vocal': round(vmean, 3),
        })
    # coalesce same-kind neighbours
    merged = []
    for sec in sections:
        if merged and merged[-1]['kind'] == sec['kind']:
            merged[-1]['end'] = sec['end']
            merged[-1]['energy'] = round((merged[-1]['energy'] + sec['energy']) / 2, 3)
        else:
            merged.append(sec)
    sections = merged

    # --- mix windows from the groove (drums+bass present), phrase-snapped ---
    groove = (drmS > DRUM_ON) & (bssS > BASS_ON)
    first_groove = int(np.argmax(groove)) if groove.any() else 0
    last_groove = nb - 1 - int(np.argmax(groove[::-1])) if groove.any() else nb - 1

    def bar_snap(beat_i):
        return float(round(phase + round(beat_i / 4) * 4 * beat_sec, 1))

    mix_in_start = 0.0
    mix_in_end = bar_snap(first_groove)
    # mix-out: a DJ starts the outro blend at a natural EXIT with a real runway — a
    # late breakdown (bass thinned, easy to blend B under), else a phrase block
    # giving ~24 bars before the end. NOT the tiny end outro (last 2 bars).
    end_runway = max(0.0, bar_snap(last_groove) - 24 * bar_sec)
    late_break = next((sec['start'] for sec in reversed(sections)
                       if sec['kind'] == 'breakdown' and dur * 0.55 < sec['start'] < dur - 8 * bar_sec), None)
    mix_out_start = float(late_break) if late_break is not None else float(end_runway)
    # never later than ~8 bars before the end (always leave a runway)
    mix_out_start = float(min(mix_out_start, max(0.0, dur - 8 * bar_sec)))

    def window(a, b):
        a, b = float(a), float(b)
        bars = int(round((b - a) / bar_sec)) if bar_sec else None
        return {'startSec': round(a, 1), 'endSec': round(b, 1), 'bars': bars}

    return {
        'dur': round(float(dur), 1),
        'barSec': round(bar_sec, 6),
        'sections': sections,
        'mixIn': window(mix_in_start, mix_in_end),
        'mixOut': window(mix_out_start, dur),
        'bass': _resample(bssS, ENV_BINS),
        'energyEnv': _resample(totS, ENV_BINS),
        'vocal': _resample(vocS, ENV_BINS),
        'drums': _resample(drmS, ENV_BINS),
        'confidence': 'full',
        'structureSource': 'demucs',
    }
