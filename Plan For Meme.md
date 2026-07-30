Ji 😈🔥
Now I’ll give you a **clear engineering-level plan** that you can directly give to Antigravity to build.

No fluff. Structured. Production-ready. GitHub Actions compatible.

---

# 🎯 PROJECT GOAL

Build a fully automated system that:

1. Scrapes raw relatable videos (Indian audience focus)
2. Downloads video
3. Cleans audio
4. Transcribes audio
5. Extracts frames
6. Performs vision analysis
7. Merges audio + vision context
8. Sends context to AI model
9. Generates:

   * Situation summary
   * Meme hook
   * Top text
   * Subtitles
   * Hashtags
10. Exports structured JSON for rendering pipeline

Runs entirely on:

* GitHub Actions (CPU only)
* Free forever
* No paid APIs (except optional LLM stage)

---

# 🏗 HIGH-LEVEL ARCHITECTURE

```
Reddit Scraper
      ↓
Video Downloader
      ↓
Audio Cleaner (FFmpeg)
      ↓
Whisper Transcription
      ↓
Frame Extraction (FFmpeg)
      ↓
BLIP Vision Captioning
      ↓
Context Builder
      ↓
AI Caption Generator
      ↓
Structured Meme Output (JSON)
```

---

# 📦 STAGE 1 — SCRAPING

### Sources:

r/IndianTeenagers
r/IndiaSocial
r/PublicFreakout
r/WatchPeopleDieInside
r/therewasanattempt
r/PublicFreakout
r/ActualPublicFreakouts
r/WatchPeopleDieInside
r/therewasanattempt
r/IdiotsInCars
r/Unexpected
r/WhatCouldGoWrong
r/ContagiousLaughter
r/IndianTeenagers
r/IndiaSocial
r/IndianDankMemes (ONLY raw uploads, not template memes)
r/DesiMemes (filter carefully)
r/IndianBoysOnTinder
r/JEENEETards (student panic gold 😂)
r/AskIndia (sometimes video links)
r/CringeIndia
r/PublicFreakout
r/WatchPeopleDieInside
r/therewasanattempt
r/ContagiousLaughter
r/Unexpected

### Filter Rules:

* Must contain video
* Duration: 5–40 sec
* No obvious watermark
* No movie/TV tags
* Prefer < 1 week old

### Output:

```json
{
  "video_url": "...",
  "reddit_title": "...",
  "reddit_sub": "...",
  "upvotes": 1200
}
```

---

# 📦 STAGE 2 — VIDEO PROCESSING

## 2.1 Download Video

Use yt-dlp or direct mp4 URL.

---

## 2.2 Audio Cleaning (IMPORTANT)

```bash
ffmpeg -i video.mp4 -af "highpass=f=200, lowpass=f=3000, dynaudnorm" cleaned.wav
```

Improves transcription quality significantly.

---

## 2.3 Whisper Transcription

Model:

```
tiny.en
```

Output:

```json
{
  "text": "...",
  "segments": [...]
}
```

If transcript < 5 words → mark as LOW_AUDIO_CONFIDENCE

---

# 📦 STAGE 3 — VISION UNDERSTANDING

## 3.1 Frame Extraction

```bash
ffmpeg -i video.mp4 -vf fps=0.5 frames/out%d.jpg
```

Limit:
Max 5 frames.

---

## 3.2 BLIP Captioning

For each frame:

Generate caption.

Store array:

```json
[
  "a young man looking embarrassed",
  "students laughing in background",
  "classroom environment"
]
```

---

# 📦 STAGE 4 — CONTEXT BUILDER

Combine:

```json
{
  "reddit_title": "...",
  "transcript": "...",
  "vision_descriptions": [...],
  "audio_confidence": "low" or "normal"
}
```

---

# 📦 STAGE 5 — AI INFERENCE PROMPT

Send this to AI model:

```
You are creating relatable meme content for Indian Gen-Z audience.

Video Title:
{{reddit_title}}

Transcript:
{{transcript}}

Scene Observations:
{{vision_descriptions}}

If transcript is weak, rely more on scene descriptions.
If both are unclear, infer most likely relatable situation.

Return structured JSON with:
1. situation_summary (1-2 lines)
2. meme_hook (short reel hook)
3. top_text (main meme text)
4. subtitle_clean (rewritten dialogue)
5. hashtags (10 Indian relatable tags)
6. emotion_detected
7. confidence_score (1-10)
```

Return JSON only.

---

# 📦 STAGE 6 — VALIDATION LOGIC (VERY IMPORTANT)

Before rendering:

Reject if:

* confidence_score < 6
* transcript empty AND vision generic
* video duration < 4 sec
* No human detected in vision

This prevents low-quality posts.

---

# ⚙️ GITHUB ACTIONS REQUIREMENTS

Must include:

### Cache HuggingFace Models

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/huggingface
    key: hf-model-cache
```

### Install Dependencies

* Python
* FFmpeg
* Torch CPU
* Transformers
* Whisper

---

# 🚨 IMPORTANT ENGINEERING CONSIDERATIONS

## 1️⃣ Memory Management

* Load BLIP once
* Process all frames
* Release model
* Then run next video

Do NOT reload model per frame.

---

## 2️⃣ Timeout Protection

GitHub Actions has 6 hour limit.

Add:
Max videos per run: 3–5

---

## 3️⃣ Fallback Strategy

If:

* Whisper fails
* BLIP fails

Mark video as FAILED and skip.

Never crash workflow.

---

## 4️⃣ Avoid Heavy LLM in GitHub

Do NOT run 7B model inside GitHub runner.

Use:

* External API
  OR
* Lightweight model via Ollama server

---

## 5️⃣ Store Structured Logs

Save JSON output per video in:

```
/output/video_id.json
```

Helps debugging.

---

# 🧠 QUALITY IMPROVEMENT IDEAS (IMPORTANT)

### Emotion Mapping

After vision captions,
detect emotion:

Keywords:

* embarrassed
* angry
* laughing
* shocked
* stressed

Map to:
Student / Relationship / Office / Social awkward

This improves meme relevance.

---

### Scene Type Classification

Basic keyword classifier:

If captions contain:

* classroom → student meme
* office → job meme
* street interview → public awkward meme

Then feed this as context to AI.

---

# 🎯 FINAL OUTPUT STRUCTURE

Each processed clip must output:

```json
{
  "video_id": "...",
  "situation_summary": "...",
  "meme_hook": "...",
  "top_text": "...",
  "subtitle_clean": "...",
  "hashtags": [...],
  "emotion": "...",
  "confidence_score": 8
}
```

Renderer then:

* Adds top_text
* Adds subtitle_clean
* Adds BGM + SFX
* Uploads

---

# 🔥 MOST IMPORTANT THINGS FOR THIS AUTOMATION

1. Model caching
2. Limit frame count
3. Limit video duration
4. Confidence scoring
5. Skip low-context videos
6. Always combine audio + vision
7. Never trust transcript alone
8. Never trust vision alone
9. Always validate AI JSON format
10. Log everything

---

Ji this is now a real production-grade meme AI pipeline.

If you want, next I can:

* Design the rendering automation plan
* Design upload automation plan
* Or design scaling system for multiple pages 😈🔥