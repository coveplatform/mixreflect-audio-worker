"""Run a local audio file through the exact worker pipeline (analyze_file →
to_audio_features + _report_waveform) and dump the JSON the app would receive.

Usage: python _analyze_local.py <audio file> <out.json> [--deep]
"""
import json
import os
import subprocess
import sys
import tempfile

import analyze
import worker


def main():
    src = sys.argv[1]
    out_path = sys.argv[2]
    deep = "--deep" in sys.argv
    work_dir = tempfile.mkdtemp(prefix="djmix-local-")
    wav = os.path.join(work_dir, "input.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "2", "-ar", "44100", wav],
        check=True, capture_output=True,
    )
    rec = analyze.analyze_file(wav, work_dir, quick=not deep)
    feat = worker.to_audio_features(rec)
    wf = worker._report_waveform(work_dir, rec.get("id"))
    if wf:
        feat["waveform"] = wf
    feat["_title"] = rec.get("title")
    feat["_bpm"] = rec.get("bpm")
    with open(out_path, "w") as f:
        json.dump(feat, f, indent=1)
    print(f"wrote {out_path}  (deep={deep}, waveform={'yes' if wf else 'MISSING'})")


if __name__ == "__main__":
    main()
