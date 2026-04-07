#!/usr/bin/env python3
"""
Original Meme Factory — Video Processing Pipeline
==================================================
Stages:
  1. Download raw clip (yt-dlp)
  2. Validate duration (5-60s)
  3. Clean audio (FFmpeg highpass/lowpass/dynaudnorm)
  4. Whisper transcription (small model, high accuracy)
  5. Frame extraction (FFmpeg, max 5 key frames)
  6. Groq Vision captioning (per frame)
  7. Emotion detection + Scene classification
  8. Context builder (merge audio + vision + metadata)
  9. AI meme text generation (Groq llama-3.3-70b)
  10. Validation gate (confidence >= 6)
  11. FFmpeg rendering (text overlay + SFX + BGM + anti-detection)
  12. Upload to Catbox
  13. Callback to n8n webhook

Designed to run on GitHub Actions (Ubuntu, CPU only, 7GB RAM).
"""

import os
import sys
import json
import subprocess
import requests
import time
import glob
import random
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")

WHISPER_MODEL = "base"          # Good accuracy, 2x faster than small on CPU
GROQ_MODEL = "llama-3.3-70b-versatile"           # Primary text gen (best quality)
GROQ_TEXT_FALLBACK = "llama-3.1-8b-instant"       # Fallback text gen (separate rate limit, higher RPM)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision

MAX_FRAMES = 3                  # 3 frames = enough context, saves API calls
MIN_CONFIDENCE = 2              # Reject memes below this score
MAX_VIDEO_DURATION = 60         # Max output duration in seconds
MIN_VIDEO_DURATION = 5          # Too short = useless
WATERMARK_TEXT = "@Meme_Facteory"

WORK_DIR = "work"
OUTPUT_DIR = "output"

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    """Main pipeline orchestrator."""
    video_url = os.environ.get("VIDEO_URL")
    video_id = os.environ.get("VIDEO_ID")
    reddit_title = os.environ.get("REDDIT_TITLE", "")
    reddit_sub = os.environ.get("REDDIT_SUB", "")
    # Pre-resolved v.redd.it URL from the proxy (for requests-based fallback)
    reddit_video_url_env = os.environ.get("REDDIT_VIDEO_URL", "").strip()

    if not video_url or not video_id:
        print("ERROR: VIDEO_URL and VIDEO_ID are required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "video_id": video_id,
        "content_type": "video",
        "title": reddit_title,
        "status": "processing",
        "stages_completed": [],
        "stages_failed": [],
    }

    try:
        # ── Stage 0: Pre-validate Reddit URL ─────────────────────────────
        # Use REDDIT_VIDEO_URL env if already provided (pre-resolved by proxy)
        resolved_video_url = reddit_video_url_env if reddit_video_url_env else None
        if "reddit.com" in video_url or "redd.it" in video_url:
            log_stage("PRE-VALIDATE URL")
            is_video, resolved_url = check_reddit_post_type(video_url)
            if not is_video:
                output["status"] = "skipped"
                output["reason"] = "Post is not a video (image/text/link post)"
                print(f"  ⏭️ SKIPPED: Not a video post")
                save_output(output, video_id)
                notify_n8n(output)
                return
            if resolved_url and not resolved_video_url:
                resolved_video_url = resolved_url
                print(f"  Resolved video URL: {resolved_video_url}")
            elif resolved_video_url:
                print(f"  Using pre-resolved video URL: {resolved_video_url}")

        # ── Stage 1: Download ─────────────────────────────────────────────
        log_stage("DOWNLOAD")
        raw_video = download_video(video_url, video_id, resolved_video_url)
        output["stages_completed"].append("download")

        # ── Stage 2: Validate Duration ────────────────────────────────────
        log_stage("VALIDATE DURATION")
        duration = get_video_duration(raw_video)
        print(f"  Video duration: {duration:.1f}s")

        if duration < MIN_VIDEO_DURATION:
            raise ValueError(f"Video too short ({duration:.1f}s < {MIN_VIDEO_DURATION}s)")

        if duration > MAX_VIDEO_DURATION + 10:
            print(f"  Trimming from {duration:.1f}s to {MAX_VIDEO_DURATION}s")
            raw_video = trim_video(raw_video, MAX_VIDEO_DURATION)

        output["stages_completed"].append("validate_duration")

        # ── Stage 3: Clean Audio ──────────────────────────────────────────
        log_stage("CLEAN AUDIO")
        cleaned_audio = clean_audio(raw_video)
        output["stages_completed"].append("audio_clean")

        # ── Stage 4: Whisper Transcription ────────────────────────────────
        log_stage("WHISPER TRANSCRIPTION")
        transcript_data = transcribe_audio(cleaned_audio)
        print(f"  Language: {transcript_data['language']}")
        print(f"  Confidence: {transcript_data['confidence']}")
        print(f"  Text: {transcript_data['text'][:200]}")
        output["stages_completed"].append("whisper")

        # ── Stage 5: Frame Extraction ─────────────────────────────────────
        log_stage("FRAME EXTRACTION")
        frames = extract_frames(raw_video)
        print(f"  Extracted {len(frames)} frames")
        output["stages_completed"].append("frame_extract")

        # ── Stage 6: Groq Vision Captioning ───────────────────────────────
        log_stage("GROQ VISION CAPTIONING")
        vision_descriptions = caption_frames(frames)
        for i, desc in enumerate(vision_descriptions):
            print(f"  Frame {i+1}: {desc}")
        output["stages_completed"].append("groq_vision")

        # ── Stage 7: Context Builder ──────────────────────────────────────
        log_stage("CONTEXT BUILDER")
        context = build_context(
            reddit_title, reddit_sub,
            transcript_data, vision_descriptions
        )
        print(f"  Emotion: {context['detected_emotion']}")
        print(f"  Scene: {context['scene_type']}")
        output["stages_completed"].append("context_build")

        # ── Stage 8: AI Meme Text Generation ──────────────────────────────
        log_stage("AI MEME GENERATION")
        meme_data = generate_meme_text(context)
        print(f"  Top text: {meme_data.get('top_text', 'N/A')}")
        print(f"  Style: {meme_data.get('meme_style', 'N/A')}")
        print(f"  SFX: {meme_data.get('sfx_suggestion', 'N/A')}")
        print(f"  Confidence: {meme_data.get('confidence_score', 'N/A')}")
        output["stages_completed"].append("ai_generate")

        # ── Stage 9: Validation Gate ──────────────────────────────────────
        log_stage("VALIDATION")
        is_valid, reject_reasons = validate_meme(meme_data, context)

        if not is_valid:
            output["status"] = "skipped"
            output["reason"] = "; ".join(reject_reasons)
            print(f"  REJECTED: {output['reason']}")
            save_output(output, video_id)
            notify_n8n(output)
            return

        print("  PASSED ✅")
        output["stages_completed"].append("validation")

        # ── Stage 10: FFmpeg Rendering ────────────────────────────────────
        log_stage("RENDER")
        final_video = render_meme(raw_video, meme_data, transcript_data, video_id)
        output["stages_completed"].append("render")

        # Verify final video
        final_duration = get_video_duration(final_video)
        final_size = os.path.getsize(final_video) / (1024 * 1024)
        print(f"  Final: {final_duration:.1f}s, {final_size:.1f}MB")

        # ── Stage 11: Upload ──────────────────────────────────────────────
        log_stage("UPLOAD")
        download_url = upload_to_catbox(final_video)
        print(f"  URL: {download_url}")
        output["stages_completed"].append("upload")

        # ── Final Output ──────────────────────────────────────────────────
        output.update({
            "status": "ready",
            "download_url": download_url,
            "caption": meme_data.get("caption", ""),
            "meme_data": meme_data,
            "duration": final_duration,
            "file_size_mb": round(final_size, 2),
        })

    except Exception as e:
        output["status"] = "failed"
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"\n❌ PIPELINE FAILED: {e}", file=sys.stderr)
        traceback.print_exc()

    # Always save output and notify
    save_output(output, video_id)
    notify_n8n(output)

    if output["status"] == "ready":
        print(f"\n✅ SUCCESS: Meme ready at {output.get('download_url')}")
    elif output["status"] == "skipped":
        print(f"\n⏭️ SKIPPED: {output.get('reason')}")
    else:
        print(f"\n❌ FAILED: {output.get('error')}")
        sys.exit(1)


# ─── Stage Functions ─────────────────────────────────────────────────────────

def check_reddit_post_type(url):
    """Pre-check: verify the Reddit URL is a video post, not an image/text post.
    
    Returns: (is_video: bool, resolved_video_url: str or None)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    # v.redd.it URLs are always video
    if "v.redd.it" in url:
        return True, url

    try:
        json_url = None
        if "reddit.com" in url:
            json_url = url.rstrip("/") + ".json"
        elif "redd.it" in url:
            json_url = f"https://www.reddit.com/comments/{url.split('/')[-1]}.json"
        
        if not json_url:
            return True, None  # Can't check, assume video

        cookie_jar = _load_cookies_for_requests()
        resp = requests.get(json_url, headers=headers, cookies=cookie_jar, timeout=15, allow_redirects=True)
        
        if resp.status_code != 200:
            print(f"  Could not verify post type (HTTP {resp.status_code}), trying anyway...")
            return True, None  # Can't check, try download anyway

        data = resp.json()

        # Navigate Reddit JSON
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
        elif isinstance(data, dict):
            post_data = data.get("data", {}).get("children", [{}])[0].get("data", {})
        else:
            return True, None

        is_video = post_data.get("is_video", False)
        post_hint = post_data.get("post_hint", "")
        domain = post_data.get("domain", "")

        print(f"  Post type: is_video={is_video}, hint={post_hint}, domain={domain}")

        # Check if it's a video
        if is_video:
            # Extract the video URL
            media = post_data.get("secure_media") or post_data.get("media")
            if media and "reddit_video" in media:
                video_url = media["reddit_video"].get("fallback_url", "")
                video_url = video_url.split("?")[0]
                return True, video_url
            return True, None

        # Check crossposts
        crosspost = post_data.get("crosspost_parent_list", [])
        if crosspost and crosspost[0].get("is_video"):
            media = crosspost[0].get("secure_media") or crosspost[0].get("media")
            if media and "reddit_video" in media:
                video_url = media["reddit_video"].get("fallback_url", "")
                video_url = video_url.split("?")[0]
                return True, video_url

        # Check for external video links (YouTube, Streamable, etc.)
        if post_hint == "rich:video" or domain in ["youtube.com", "youtu.be", "streamable.com"]:
            ext_url = post_data.get("url_overridden_by_dest") or post_data.get("url")
            return True, ext_url

        # Not a video
        if post_hint == "image":
            print(f"  ❌ This is an IMAGE post, not a video")
        elif post_hint == "self":
            print(f"  ❌ This is a TEXT post, not a video")
        elif post_hint == "link":
            print(f"  ❌ This is a LINK post, not a video")
        else:
            print(f"  ❌ Post type '{post_hint}' is not a video")

        return False, None

    except Exception as e:
        print(f"  Warning: Could not verify post type ({e}), trying download anyway...")
        return True, None  # On error, try download anyway

def download_video(url, video_id, resolved_video_url=None):
    """Stage 1: Download video using yt-dlp with Reddit fallbacks.
    
    Args:
        url: Original URL (Reddit post URL or direct URL)
        video_id: Video ID for filename
        resolved_video_url: Pre-resolved v.redd.it URL for fallback downloads
    """
    output_path = f"{WORK_DIR}/{video_id}_raw.mp4"

    # Check for cookies file
    cookies_arg = []
    has_cookies = os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 0
    if has_cookies:
        cookies_arg = ["--cookies", "cookies.txt"]
        print(f"  Cookies file loaded ({os.path.getsize('cookies.txt')} bytes)")
    else:
        print(f"  ⚠️ No cookies.txt found — Reddit downloads may fail!")

    # Load cookies for requests-based downloads
    cookie_jar = _load_cookies_for_requests()

    # Common headers for all requests
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "video/*,*/*",
        "Referer": "https://www.reddit.com/",
    }

    # ── Try yt-dlp first (with original Reddit post URL for [Reddit] extractor) ──
    print(f"  yt-dlp attempt with URL: {url}")
    common_args = [
        "--user-agent", HEADERS["User-Agent"],
        "--referer", "https://www.reddit.com/",
        "--geo-bypass",
        "--no-check-certificates",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
        "-o", output_path,
    ]

    cmd = [
        "yt-dlp", *cookies_arg,
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/b",
        "--merge-output-format", "mp4",
        *common_args, url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return _validate_and_return(output_path, video_id)

    print(f"  yt-dlp failed ({result.stderr.strip()[:100]}), trying fallbacks...")

    # ── Use pre-resolved v.redd.it URL if available ────────────────────────
    fallback_url = resolved_video_url

    # If no pre-resolved URL, try Reddit JSON API to resolve it
    if not fallback_url:
        fallback_url = _resolve_reddit_video_url(url, HEADERS)

    if fallback_url and "v.redd.it" in fallback_url:
        print(f"  Trying direct download: {fallback_url}")
        success = _download_with_requests(fallback_url, output_path, HEADERS, cookie_jar)
        if success:
            return _validate_and_return(output_path, video_id)

        # Try DASH quality cascade
        base_url = fallback_url.split("/DASH")[0] if "/DASH" in fallback_url else fallback_url.rstrip("/")
        dash_qualities = ["DASH_720.mp4", "DASH_480.mp4", "DASH_360.mp4", "DASH_240.mp4"]

        for quality in dash_qualities:
            dash_url = f"{base_url}/{quality}"
            print(f"  Trying DASH: {quality}...")
            success = _download_with_requests(dash_url, output_path, HEADERS, cookie_jar)
            if success:
                return _validate_and_return(output_path, video_id)

    # ── Last resort: yt-dlp with no format selection ──────────────────────
    cmd_auto = ["yt-dlp", *cookies_arg, *common_args, url]
    result = subprocess.run(cmd_auto, capture_output=True, text=True, timeout=300)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return _validate_and_return(output_path, video_id)

    raise RuntimeError(
        f"Download failed (all attempts). URL: {url}\n"
        f"yt-dlp stderr: {result.stderr[-300:]}"
    )


def _load_cookies_for_requests():
    """Load cookies from cookies.txt (Netscape format) for use with requests."""
    cookies = {}
    if not os.path.exists("cookies.txt"):
        return cookies
    try:
        with open("cookies.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    cookies[parts[5]] = parts[6]
        print(f"  Loaded {len(cookies)} cookies from cookies.txt")
    except Exception as e:
        print(f"  Warning: Could not parse cookies.txt: {e}")
    return cookies


def _resolve_reddit_video_url(url, headers):
    """Use Reddit's JSON API to extract the actual video download URL."""
    try:
        # Convert Reddit post URL to JSON API URL
        json_url = None

        if "reddit.com" in url:
            # Post URL like https://www.reddit.com/r/funny/comments/abc123/...
            json_url = url.rstrip("/") + ".json"
        elif "redd.it" in url and "v.redd.it" not in url:
            # Short URL like https://redd.it/abc123
            json_url = f"https://www.reddit.com/comments/{url.split('/')[-1]}.json"
        else:
            # v.redd.it URL - try to use it directly
            return url

        if not json_url:
            return None

        # Load cookies for authenticated Reddit access
        cookie_jar = _load_cookies_for_requests()

        print(f"  Resolving via Reddit JSON API...")
        resp = requests.get(json_url, headers=headers, cookies=cookie_jar, timeout=15, allow_redirects=True)

        if resp.status_code != 200:
            print(f"  Reddit JSON API returned {resp.status_code}")
            return None

        data = resp.json()

        # Navigate the Reddit JSON structure
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
        elif isinstance(data, dict):
            post_data = data.get("data", {}).get("children", [{}])[0].get("data", {})
        else:
            return None

        # Try to get the video URL from the Reddit media object
        media = post_data.get("secure_media") or post_data.get("media")
        if media and "reddit_video" in media:
            video_url = media["reddit_video"].get("fallback_url")
            if video_url:
                # Remove query params that might cause issues
                video_url = video_url.split("?")[0]
                print(f"  Resolved video URL: {video_url}")
                return video_url

        # Try crosspost parent
        crosspost = post_data.get("crosspost_parent_list", [])
        if crosspost:
            media = crosspost[0].get("secure_media") or crosspost[0].get("media")
            if media and "reddit_video" in media:
                video_url = media["reddit_video"].get("fallback_url")
                if video_url:
                    video_url = video_url.split("?")[0]
                    print(f"  Resolved crosspost video URL: {video_url}")
                    return video_url

        # Try url_overridden_by_dest (external video links)
        external_url = post_data.get("url_overridden_by_dest") or post_data.get("url")
        if external_url and any(ext in external_url for ext in [".mp4", ".webm", "v.redd.it"]):
            print(f"  Found external URL: {external_url}")
            return external_url

        print(f"  Could not resolve video URL from Reddit JSON")
        return None

    except Exception as e:
        print(f"  Reddit JSON API error: {e}")
        return None


def _download_with_requests(url, output_path, headers, cookies=None):
    """Download a file using requests with proper headers + cookies. Returns True on success."""
    try:
        resp = requests.get(url, headers=headers, cookies=cookies, stream=True, timeout=120, allow_redirects=True)

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {url}")
            return False

        # Check content-type isn't HTML (error page)
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            print(f"  Got HTML instead of video (blocked)")
            return False

        # Download the file
        total_size = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total_size += len(chunk)

        # Check minimum file size (10KB = likely not a real video)
        if total_size < 10000:
            print(f"  Downloaded file too small ({total_size} bytes), likely not a video")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        print(f"  Downloaded {total_size / 1024 / 1024:.1f}MB")
        return True

    except Exception as e:
        print(f"  Download error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def _validate_and_return(output_path, video_id):
    """Validate the downloaded file is a real video and return the path."""
    if not os.path.exists(output_path):
        # yt-dlp might have used a different extension
        for ext in [".mp4", ".webm", ".mkv"]:
            alt = f"{WORK_DIR}/{video_id}_raw{ext}"
            if os.path.exists(alt):
                if ext != ".mp4":
                    subprocess.run([
                        "ffmpeg", "-y", "-i", alt,
                        "-c:v", "libx264", "-c:a", "aac", output_path,
                    ], capture_output=True, check=True)
                    os.remove(alt)
                else:
                    output_path = alt
                break
        else:
            raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    # Verify it's a real video file
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", output_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        # Check what we actually downloaded
        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        # Read first bytes to check if it's HTML
        with open(output_path, "rb") as f:
            head = f.read(200)
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            os.remove(output_path)
            raise RuntimeError("Downloaded an HTML error page instead of video (Reddit blocked)")
        raise RuntimeError(f"Downloaded file is not a valid video (size: {file_size} bytes)")

    print(f"  Downloaded: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f}MB)")
    return output_path


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def trim_video(video_path, max_duration):
    """Trim video to max duration without re-encoding."""
    trimmed = video_path.replace("_raw.", "_trimmed.")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(max_duration), "-c", "copy", trimmed,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return trimmed


def clean_audio(video_path):
    """Stage 3: Clean audio for better Whisper accuracy.

    - highpass=200Hz: removes low rumble/wind/traffic noise
    - lowpass=3000Hz: removes high-pitched hiss/electronic noise
    - dynaudnorm: normalizes volume levels across the clip
    - Resampled to 16kHz mono (Whisper's expected input)
    """
    output = f"{WORK_DIR}/cleaned_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-af", "highpass=f=200,lowpass=f=3000,dynaudnorm=p=0.9:s=5",
        "-ar", "16000", "-ac", "1",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: extract audio without cleaning
        print("  WARNING: Audio cleaning failed, extracting raw audio")
        cmd_fallback = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", output,
        ]
        subprocess.run(cmd_fallback, capture_output=True, check=True)

    return output


def transcribe_audio(audio_path):
    """Stage 4: Transcribe using Whisper 'small' model.

    - 'small' model: 244M params, excellent accuracy for Hinglish/English
    - Auto-detects language
    - Returns text + timed segments for subtitle generation
    """
    import whisper

    print(f"  Loading Whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)

    print("  Transcribing...")
    result = model.transcribe(
        audio_path,
        language=None,      # Auto-detect language
        task="transcribe",
        verbose=False,
        fp16=False,          # CPU mode
        condition_on_previous_text=True,
    )

    # Free GPU/CPU memory immediately
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    import gc
    gc.collect()

    transcript = result.get("text", "").strip()
    segments = result.get("segments", [])
    detected_language = result.get("language", "unknown")

    # Assess confidence
    confidence = "normal"
    word_count = len(transcript.split())
    if word_count < 3:
        confidence = "very_low"
    elif word_count < 8:
        confidence = "low"

    return {
        "text": transcript,
        "segments": [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip(),
            }
            for s in segments
        ],
        "language": detected_language,
        "confidence": confidence,
        "word_count": word_count,
    }


def extract_frames(video_path, max_frames=MAX_FRAMES):
    """Stage 5: Extract key frames for vision analysis.

    Uses scene-change detection to get the most informative frames,
    falling back to uniform sampling if scene detection yields too few.
    """
    frames_dir = f"{WORK_DIR}/frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Try scene-change detection first (gets more interesting frames)
    cmd_scene = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='gt(scene,0.3)',scale=640:-1",
        "-frames:v", str(max_frames),
        "-vsync", "vfr",
        "-q:v", "2",
        f"{frames_dir}/frame_%03d.jpg",
    ]
    subprocess.run(cmd_scene, capture_output=True)

    frames = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    # If scene detection gave < 3 frames, fall back to uniform sampling
    if len(frames) < 3:
        # Clear and re-extract with uniform fps
        for f in frames:
            os.remove(f)

        duration = get_video_duration(video_path)
        fps = max(0.5, max_frames / duration)  # At least 0.5 fps

        cmd_uniform = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps:.2f},scale=640:-1",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            f"{frames_dir}/frame_%03d.jpg",
        ]
        subprocess.run(cmd_uniform, capture_output=True, check=True)
        frames = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    return frames[:max_frames]


def caption_frames(frame_paths):
    """Stage 6: Describe frames using Groq Vision API.

    Uses Groq llama-4-scout-17b-16e-instruct for fast, accurate frame analysis.
    """
    if not frame_paths:
        return ["no frames extracted"]

    import base64
    import io
    from PIL import Image

    if not OPENROUTER_API_KEY and not GROQ_API_KEY:
        return ["No vision API key set — cannot caption frames"]

    captions = []
    for i, frame_path in enumerate(frame_paths):
        try:
            # Load and resize frame for fast API processing
            img = Image.open(frame_path).convert("RGB")
            max_dim = 512
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this video frame in 2-3 sentences. Include: what's happening, who's visible, their emotions/expressions, and any TEXT visible on screen (read and quote it exactly). If it's a screenshot of a conversation/tweet/post, transcribe the full text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                }
            ]

            caption = None

            # Direct Groq Vision call (only reliable provider)
            if GROQ_API_KEY:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": GROQ_VISION_MODEL,
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.3,
                }
                try:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers, json=payload, timeout=30
                    )
                    if resp.status_code == 429:
                        wait = min(int(resp.headers.get("retry-after", 10)), 20)
                        print(f"  Groq rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        resp = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers, json=payload, timeout=30
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content")
                        if content:
                            caption = content.strip()
                            print(f"  [Groq] ✓ Frame {i+1}: {caption}")
                    else:
                        print(f"  Groq returned {resp.status_code}: {resp.text[:150]}")
                except Exception as e:
                    print(f"  Groq vision failed: {e}")

            captions.append(caption or "could not caption frame")

            # Short delay between frames to avoid rate limits
            if i < len(frame_paths) - 1:
                time.sleep(1)

        except Exception as e:
            print(f"  WARNING: Frame {i+1} captioning failed ({e})")
            captions.append("unclear frame content")

    return captions


# ─── Context Analysis ────────────────────────────────────────────────────────

def detect_emotion(vision_descriptions, transcript):
    """Detect dominant emotion from combined context."""
    text = (" ".join(vision_descriptions) + " " + transcript).lower()

    emotion_keywords = {
        "embarrassed": [
            "embarrass", "shy", "awkward", "caught", "hiding", "cringe",
            "blush", "shame", "oops", "mistake",
        ],
        "angry": [
            "angry", "shout", "fight", "argue", "furious", "yell",
            "scream", "rage", "mad", "slap",
        ],
        "happy": [
            "laugh", "smile", "dance", "happy", "joy", "celebrat",
            "cheer", "grin", "excited", "fun",
        ],
        "shocked": [
            "shock", "surprise", "jaw", "disbelief", "unexpected",
            "omg", "gasp", "stun", "amaz", "wtf",
        ],
        "stressed": [
            "stress", "cry", "panic", "anxious", "worry", "nervous",
            "exam", "deadline", "fear", "sweat",
        ],
        "confused": [
            "confus", "lost", "puzzl", "what", "huh", "scratch",
            "wonder", "think",
        ],
        "cringe": [
            "cringe", "second hand", "painful", "uncomfortable",
            "die inside", "facepalm",
        ],
    }

    scores = {}
    for emotion, keywords in emotion_keywords.items():
        scores[emotion] = sum(1 for k in keywords if k in text)

    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def classify_scene(vision_descriptions, transcript):
    """Classify scene type from context."""
    text = (" ".join(vision_descriptions) + " " + transcript).lower()

    scene_keywords = {
        "classroom": [
            "class", "student", "teacher", "desk", "board", "school",
            "lecture", "exam", "notebook", "backpack",
        ],
        "office": [
            "office", "laptop", "meeting", "boss", "work", "corporate",
            "desk", "computer", "presentation",
        ],
        "street": [
            "street", "road", "car", "traffic", "walk", "outdoor",
            "bike", "scooter", "rickshaw", "highway",
        ],
        "relationship": [
            "girl", "boy", "couple", "date", "crush", "love",
            "hug", "kiss", "flirt", "romantic",
        ],
        "home": [
            "kitchen", "room", "bed", "home", "family", "parent",
            "mom", "dad", "sibling", "couch", "sofa",
        ],
        "party": [
            "party", "club", "music", "dance", "crowd", "concert",
            "dj", "wedding", "celebration",
        ],
        "food": [
            "food", "eat", "restaurant", "cook", "meal", "plate",
            "spicy", "chai", "biryani",
        ],
    }

    scores = {}
    for scene, keywords in scene_keywords.items():
        scores[scene] = sum(1 for k in keywords if k in text)

    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def build_context(reddit_title, reddit_sub, transcript_data, vision_descriptions):
    """Stage 7: Build unified context from all sources."""
    emotion = detect_emotion(vision_descriptions, transcript_data["text"])
    scene_type = classify_scene(vision_descriptions, transcript_data["text"])

    return {
        "reddit_title": reddit_title,
        "reddit_sub": reddit_sub,
        "transcript": transcript_data["text"],
        "transcript_segments": transcript_data["segments"],
        "transcript_language": transcript_data["language"],
        "audio_confidence": transcript_data["confidence"],
        "word_count": transcript_data["word_count"],
        "vision_descriptions": vision_descriptions,
        "detected_emotion": emotion,
        "scene_type": scene_type,
    }


# ─── AI Meme Generation ─────────────────────────────────────────────────────

def generate_meme_text(context):
    """Stage 8: Generate meme text using Groq AI."""
    if not GROQ_API_KEY:
        raise RuntimeError("Need GROQ_API_KEY for text generation")

    # Gather available SFX names for the AI
    sfx_dir = "assets/sfx"
    available_sfx = []
    if os.path.isdir(sfx_dir):
        available_sfx = [os.path.splitext(f)[0] for f in os.listdir(sfx_dir) if f.endswith('.mp3')]

    sfx_list_str = ", ".join(available_sfx[:40])

    # Build a clear summary of what the video actually shows
    transcript = context['transcript'].strip()
    visuals = "\n".join(f"  Frame {i+1}: {v}" for i, v in enumerate(context['vision_descriptions']))

    prompt = f"""You are India's #1 viral meme creator for Gen-Z Instagram/Reels.

═══ STEP 1: UNDERSTAND THE VIDEO ═══

FIRST, carefully analyze the video content from these sources:

Reddit Title: "{context['reddit_title']}"
Subreddit: r/{context['reddit_sub']}

What the video SHOWS (frame by frame):
{visuals}

What is SAID in the video (transcript, {context['transcript_language']}, confidence: {context['audio_confidence']}):
"{transcript if transcript else 'No speech detected'}"

Detected Emotion: {context['detected_emotion']}
Scene Type: {context['scene_type']}

═══ STEP 2: CREATE A MEME ABOUT THIS EXACT VIDEO ═══

CRITICAL RULES:
1. Your meme text MUST be about what's actually happening in THIS video
2. DO NOT generate random/generic meme text — it must match the video content
3. First write situation_summary describing what happens in the video
4. Then write top_text that directly references THAT situation
5. If transcript has dialogue → use "subtitle" style with Hinglish translation
6. If transcript is empty/weak → use visual descriptions to understand the video
7. top_text: 4-6 words in Hinglish, describing THIS video's situation
   GOOD: "Jab teacher unexpectedly roll call le le" (matches a classroom video)
   BAD: Random generic text unrelated to what's shown
8. subtitle_clean: Hinglish dialogue, 2-3 words per line, separated by newlines
9. Pick 3-4 sound effects from: {sfx_list_str}

═══ RETURN FORMAT ═══

Return ONLY valid JSON, no markdown, no explanation:
{{
  "situation_summary": "1-2 lines describing what ACTUALLY happens in this video",
  "meme_hook": "Short hook that matches the video (max 10 words)",
  "top_text": "4-6 word Hinglish text about THIS video's situation",
  "bottom_text": "Optional punchline or empty string",
  "subtitle_clean": "Hinglish dialogue, 2-3 words per line, newline separated",
  "hashtags": ["10", "relevant", "hashtags"],
  "emotion_detected": "{context['detected_emotion']}",
  "confidence_score": 8,
  "sfx_suggestions": [
    {{"sfx": "sfx-name", "at_percent": 15}},
    {{"sfx": "sfx-name", "at_percent": 45}},
    {{"sfx": "sfx-name", "at_percent": 75}},
    {{"sfx": "sfx-name", "at_percent": 92}}
  ],
  "meme_style": "pov|top_bottom|subtitle|caption",
  "caption": "Instagram caption — Hinglish hook + emojis + CTA + hashtags"
}}"""

    system_msg = (
        "You are India's top meme page admin. You create viral Hinglish memes "
        "that get millions of views. You understand Indian culture, Gen-Z humor, "
        "and what makes content relatable. Return ONLY valid JSON."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    # ── Primary: Groq 70B (best quality) ──────────────────────────────────
    if GROQ_API_KEY:
        print(f"  Trying Groq 70B ({GROQ_MODEL})...")
        result = _call_text_gen_api(
            api_url="https://api.groq.com/openai/v1/chat/completions",
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            messages=messages,
            provider="Groq-70B",
        )
        if result:
            return result

    # ── Fallback: Groq 8B (separate rate limit, higher RPM) ──────────────
    if GROQ_API_KEY:
        print(f"  70B rate limited, trying 8B fallback ({GROQ_TEXT_FALLBACK})...")
        result = _call_text_gen_api(
            api_url="https://api.groq.com/openai/v1/chat/completions",
            api_key=GROQ_API_KEY,
            model=GROQ_TEXT_FALLBACK,
            messages=messages,
            provider="Groq-8B",
        )
        if result:
            return result

    # ── Last resort: retry 70B with wait ─────────────────────────────────
    if GROQ_API_KEY:
        for attempt in range(2):
            wait = 45 * (attempt + 1)  # 45s, 90s
            print(f"  Both models rate limited, waiting {wait}s...")
            time.sleep(wait)
            result = _call_text_gen_api(
                api_url="https://api.groq.com/openai/v1/chat/completions",
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                messages=messages,
                provider="Groq-70B",
            )
            if result:
                return result

    raise RuntimeError("All text generation failed")


def _call_text_gen_api(api_url, api_key, model, messages, provider="API"):
    """Try ONE text generation API call. Returns parsed meme JSON or None."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter" in api_url:
        headers["HTTP-Referer"] = "https://github.com/VikashMeena777/Meme-MKR"
        headers["X-Title"] = "Original Meme Factory"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.85,
        "max_tokens": 600,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            print(f"  {provider} rate limited")
            return None

        if response.status_code != 200:
            print(f"  {provider} returned {response.status_code}: {response.text[:150]}")
            return None

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            print(f"  {provider} returned empty content")
            return None

        # Clean JSON from markdown wrapping
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        import re
        cleaned_lines = []
        for cline in content.split('\n'):
            cleaned_lines.append(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cline))
        content = '\n'.join(cleaned_lines)

        meme_data = json.loads(content, strict=False)

        required = ["top_text", "meme_style", "confidence_score", "caption"]
        for field in required:
            if field not in meme_data:
                raise ValueError(f"Missing required field: {field}")

        print(f"  [{provider}] ✓ Meme generated")
        return meme_data

    except json.JSONDecodeError as e:
        print(f"  {provider} JSON parse failed: {e}")
    except Exception as e:
        print(f"  {provider} failed: {e}")

    return None


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_meme(meme_data, context):
    """Stage 9: Validate meme quality before rendering.

    Returns (is_valid: bool, reasons: list[str])
    """
    reasons = []

    # Check confidence
    score = meme_data.get("confidence_score", 0)
    if isinstance(score, str):
        try:
            score = int(score)
        except:
            score = 0

    if score < MIN_CONFIDENCE:
        reasons.append(f"Confidence too low ({score} < {MIN_CONFIDENCE})")

    # Check essential fields
    if not meme_data.get("top_text", "").strip():
        reasons.append("No top_text generated")

    if not meme_data.get("caption", "").strip():
        reasons.append("No caption generated")

    # Check if both audio and vision are useless
    audio_useless = context["audio_confidence"] in ("low", "very_low")
    vision_useless = all(
        any(x in desc.lower() for x in ["unclear", "blurry", "dark", "nothing"])
        for desc in context["vision_descriptions"]
    )
    if audio_useless and vision_useless:
        reasons.append("Both audio and vision are unclear/unusable")

    # Check top_text length
    if len(meme_data.get("top_text", "")) > 100:
        reasons.append("top_text too long (> 100 chars)")

    return (len(reasons) == 0, reasons)


# ─── Rendering ───────────────────────────────────────────────────────────────

def generate_srt(segments, output_path):
    """Generate SRT subtitle file from Whisper segments.
    
    Splits each segment into 2-3 word chunks for punchy, 
    short-burst subtitle display (TikTok/Reels style).
    """
    srt_index = 1
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if not text:
                continue
            
            words = text.split()
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start
            
            if len(words) <= 3:
                # Already short enough
                start_ts = format_timestamp(seg_start)
                end_ts = format_timestamp(seg_end)
                f.write(f"{srt_index}\n{start_ts} --> {end_ts}\n{text}\n\n")
                srt_index += 1
            else:
                # Split into 2-3 word chunks with proportional timing
                chunk_size = 2 if len(words) <= 6 else 3
                chunks = []
                for j in range(0, len(words), chunk_size):
                    chunks.append(" ".join(words[j:j+chunk_size]))
                
                chunk_duration = seg_duration / len(chunks) if chunks else seg_duration
                
                for k, chunk in enumerate(chunks):
                    c_start = seg_start + k * chunk_duration
                    c_end = min(c_start + chunk_duration, seg_end)
                    start_ts = format_timestamp(c_start)
                    end_ts = format_timestamp(c_end)
                    f.write(f"{srt_index}\n{start_ts} --> {end_ts}\n{chunk}\n\n")
                    srt_index += 1


def format_timestamp(seconds):
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def escape_ffmpeg_text(text):
    """Escape text for FFmpeg drawtext filter."""
    # Must escape these characters for FFmpeg
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace(";", "\\;")
    return text


def render_meme(video_path, meme_data, transcript_data, video_id):
    """Stage 10: Render final meme video with all overlays.

    Pipeline: text overlay → subtitles → SFX → BGM → anti-detection → scale → watermark
    """
    top_text = meme_data.get("top_text", "")
    bottom_text = meme_data.get("bottom_text", "")
    meme_style = meme_data.get("meme_style", "pov")
    sfx_type = meme_data.get("sfx_suggestion", "none")
    subtitle_text = meme_data.get("subtitle_clean", "")

    current_input = video_path

    # ── Step 1: Text Overlay ──────────────────────────────────────────────
    step1_output = f"{WORK_DIR}/step1_text.mp4"
    text_filter = build_text_filter(top_text, bottom_text, meme_style)

    if text_filter:
        run_ffmpeg([
            "-i", current_input,
            "-vf", text_filter,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "copy",
            step1_output,
        ])
        current_input = step1_output

    # ── Step 2: Subtitles (from Whisper segments) ─────────────────────────
    if transcript_data.get("segments") and transcript_data["confidence"] != "very_low":
        srt_path = f"{WORK_DIR}/subtitles.srt"

        if meme_style == "subtitle" and subtitle_text:
            # Use AI-rewritten subtitle
            generate_srt(transcript_data["segments"], srt_path)
        else:
            # Use Whisper segments as-is
            generate_srt(transcript_data["segments"], srt_path)

        if os.path.exists(srt_path) and os.path.getsize(srt_path) > 10:
            step2_output = f"{WORK_DIR}/step2_subs.mp4"
            subtitle_filter = (
                f"subtitles={srt_path}:force_style='"
                "FontName=Noto Sans,"
                "FontSize=10,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=3,"
                "Outline=1,"
                "Shadow=1,"
                "Alignment=2,"
                "MarginV=25'"
            )
            run_ffmpeg([
                "-i", current_input,
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "copy",
                step2_output,
            ])
            current_input = step2_output

    # ── Step 3: Sound Effects (Multiple, Context-Aware) ────────────────────
    sfx_suggestions = meme_data.get("sfx_suggestions", [])
    # Backward compat: old single sfx_suggestion field
    if not sfx_suggestions and sfx_type != "none":
        sfx_suggestions = [{"sfx": sfx_type, "at_percent": 90}]

    # Filter to only SFX files that actually exist
    valid_sfx = []
    for s in sfx_suggestions:
        sfx_name = s.get("sfx", "").strip()
        sfx_file = find_asset(f"assets/sfx/{sfx_name}.mp3")
        if sfx_file:
            valid_sfx.append({"file": sfx_file, "at_percent": s.get("at_percent", 50)})

    if valid_sfx:
        step3_output = f"{WORK_DIR}/step3_sfx.mp4"
        duration = get_video_duration(current_input)

        # Build FFmpeg inputs and filter_complex for multiple SFX
        sfx_inputs = []
        filter_parts = []
        mix_labels = ["[0:a]"]

        for idx, sfx in enumerate(valid_sfx, 1):
            sfx_inputs.extend(["-i", sfx["file"]])
            delay_ms = int(max(0, (duration * sfx["at_percent"] / 100)) * 1000)
            label = f"sfx{idx}"
            filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume=0.8[{label}]")
            mix_labels.append(f"[{label}]")

        mix_input = "".join(mix_labels)
        filter_parts.append(f"{mix_input}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=2[aout]")
        filter_complex = ";".join(filter_parts)

        run_ffmpeg([
            "-i", current_input,
            *sfx_inputs,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            step3_output,
        ])
        current_input = step3_output

    # ── Step 4: Background Music ──────────────────────────────────────────
    bgm_files = sorted(glob.glob("assets/bgm/*.mp3"))
    if bgm_files:
        bgm_file = random.choice(bgm_files)
        step4_output = f"{WORK_DIR}/step4_bgm.mp4"
        duration = get_video_duration(current_input)

        # Fade out BGM 3 seconds before end
        fade_start = max(0, duration - 4)

        run_ffmpeg([
            "-i", current_input,
            "-i", bgm_file,
            "-filter_complex",
            f"[0:a]volume=1.0[main];"
            f"[1:a]volume=0.13,afade=t=in:d=2,afade=t=out:st={fade_start:.1f}:d=3[bgm];"
            f"[main][bgm]amix=inputs=2:duration=shortest:dropout_transition=3[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            step4_output,
        ])
        current_input = step4_output

    # ── Step 5: Anti-Detection + Scale 9:16 + Watermark ───────────────────
    final_output = f"{WORK_DIR}/{video_id}_final.mp4"

    # Randomize subtle visual shifts to defeat content matching
    brightness = round(random.uniform(0.02, 0.06), 3)
    contrast = round(random.uniform(1.01, 1.05), 3)
    saturation = round(random.uniform(1.02, 1.08), 3)
    # Slight random crop (1-3% from each edge)
    crop_factor = round(random.uniform(0.96, 0.99), 3)

    watermark_escaped = escape_ffmpeg_text(WATERMARK_TEXT)

    vf_chain = (
        f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation},"
        f"crop=in_w*{crop_factor}:in_h*{crop_factor},"
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"drawtext=text='{watermark_escaped}':"
        f"fontfile={_find_font()}:"
        "fontcolor=white@0.7:fontsize=28:"
        "x=(w-text_w)/2:y=h-th-60:"
        "shadowcolor=black@0.5:shadowx=2:shadowy=2"
    )

    run_ffmpeg([
        "-i", current_input,
        "-vf", vf_chain,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-t", str(MAX_VIDEO_DURATION),
        "-movflags", "+faststart",  # Web-optimized MP4
        final_output,
    ])

    return final_output


def _find_font():
    """Find the best available font for text overlays (Noto Sans preferred for Hindi support)."""
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Last resort: let ffmpeg auto-detect
    return ""


def build_text_filter(top_text, bottom_text, style):
    """Build FFmpeg drawtext filter based on meme style.
    
    Auto-wraps long text to fit within video boundaries using FFmpeg's
    text wrapping. Font size scales with video resolution.
    """
    if not top_text:
        return ""

    font_path = _find_font()
    font = f"fontfile={font_path}:" if font_path else ""

    # Helper: wrap long text by inserting newlines (~25 chars per line for readability)
    def wrap_text(text, max_chars=25):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > max_chars:
                lines.append(current)
                current = word
            else:
                current = current + " " + word if current else word
        if current:
            lines.append(current)
        return "\n".join(lines) if lines else text

    if style == "pov":
        # POV text at top center with shadow — wrap long text
        wrapped = escape_ffmpeg_text(wrap_text(top_text, 30))
        return (
            f"drawtext=text='{wrapped}':"
            f"{font}"
            "fontsize='min(28,w/25)':"
            "fontcolor=white:"
            "borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=max(20\\,h*0.05):"
            "shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    elif style == "top_bottom":
        # Classic meme: text at top AND bottom — wrap both
        top_wrapped = escape_ffmpeg_text(wrap_text(top_text, 25))
        top_filter = (
            f"drawtext=text='{top_wrapped}':"
            f"{font}"
            "fontsize='min(32,w/22)':"
            "fontcolor=white:"
            "borderw=4:bordercolor=black:"
            "x=(w-text_w)/2:y=max(20\\,h*0.03)"
        )
        if bottom_text:
            bottom_wrapped = escape_ffmpeg_text(wrap_text(bottom_text, 25))
            top_filter += (
                f",drawtext=text='{bottom_wrapped}':"
                f"{font}"
                "fontsize='min(32,w/22)':"
                "fontcolor=white:"
                "borderw=4:bordercolor=black:"
                "x=(w-text_w)/2:y=h-th-max(20\\,h*0.03)"
            )
        return top_filter

    elif style == "caption":
        # Caption box style — text in a semi-transparent bar at top, auto-height
        wrapped = escape_ffmpeg_text(wrap_text(top_text, 35))
        return (
            "drawbox=x=0:y=0:w=iw:h=max(80\\,th+40):color=black@0.6:t=fill,"
            f"drawtext=text='{wrapped}':"
            f"{font}"
            "fontsize='min(24,w/30)':"
            "fontcolor=white:"
            "x=(w-text_w)/2:y=20"
        )

    else:
        # Default: subtitle style at bottom — wrap for readability
        wrapped = escape_ffmpeg_text(wrap_text(top_text, 28))
        return (
            f"drawtext=text='{wrapped}':"
            f"{font}"
            "fontsize='min(26,w/28)':"
            "fontcolor=white:"
            "borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=h-th-max(40\\,h*0.06):"
            "shadowcolor=black@0.5:shadowx=2:shadowy=2"
        )


# ─── Upload ──────────────────────────────────────────────────────────────────

def upload_to_catbox(video_path):
    """Stage 11: Upload to temporary file hosting."""
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Uploading {file_size:.1f}MB...")

    # Try litterbox first (72h temp hosting)
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload", "time": "72h"},
                timeout=180,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Litterbox failed: {e}")

    # Fallback to permanent catbox
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload"},
                timeout=180,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Catbox failed: {e}")

    raise RuntimeError("All upload methods failed")


# ─── Utilities ───────────────────────────────────────────────────────────────

def run_ffmpeg(args):
    """Run FFmpeg command with error handling."""
    cmd = ["ffmpeg", "-y"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Log the error but try to extract useful info
        stderr = result.stderr[-1000:] if result.stderr else "No stderr"
        raise RuntimeError(f"FFmpeg failed: {stderr}")


def find_asset(relative_path):
    """Find an asset file, checking both repo root and script directory."""
    # Check relative to repo root
    if os.path.exists(relative_path):
        return relative_path
    # Check relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, "..", relative_path)
    if os.path.exists(alt_path):
        return alt_path
    return None


def log_stage(name):
    """Print a visible stage header."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")


def save_output(data, video_id):
    """Save structured output JSON for debugging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{video_id}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Output saved: {output_path}")


def notify_n8n(data):
    """Send callback to n8n webhook."""
    if not N8N_WEBHOOK_URL:
        print("  WARNING: N8N_WEBHOOK_URL not set, skipping callback")
        return

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        print(f"  n8n webhook: {response.status_code}")
    except Exception as e:
        print(f"  WARNING: n8n callback failed: {e}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.time()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         ORIGINAL MEME FACTORY — Processing Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Whisper model: {WHISPER_MODEL}")
    print(f"  Vision: Groq {GROQ_VISION_MODEL.split('/')[-1]}")
    print(f"  Text gen: Groq {GROQ_MODEL} (fallback: {GROQ_TEXT_FALLBACK})")
    print(f"  Min confidence: {MIN_CONFIDENCE}")

    main()

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
