# djmix audio worker — DSP grounding for mixreflect score reports.
# Plain http.server + numpy/soundfile/mutagen/yt-dlp; ffmpeg installed via apt.
# Demucs/torch are intentionally NOT installed (stems.py degrades gracefully).
FROM python:3.11-slim

# ffmpeg is required (transcode downloads to wav). git helps yt-dlp on some sources.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-worker.txt .
RUN pip install --no-cache-dir -r requirements-worker.txt

COPY . .

# Stems run only on DEEP requests (quick=False), and the heavy GPU separation is
# offloaded to Replicate (set REPLICATE_API_TOKEN) — so no torch is installed and
# instant reads stay fast. DJMIX_STEMS=1 just allows the deep path to use them;
# without REPLICATE_API_TOKEN the worker falls back to loudness structure anyway.
ENV DJMIX_STEMS=1
# worker.py reads $PORT (Render/Railway/Fly inject it); 8090 is the local default.
ENV PORT=8090
EXPOSE 8090

CMD ["python", "worker.py"]
