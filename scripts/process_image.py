#!/usr/bin/env python3
"""
Original Meme Factory — Image Processing Pipeline
==================================================
Stages:
  1. Download Reddit image (requests)
  2. Groq Vision captioning (single image via API)
  3. Context analysis (emotion + scene detection)
  4. AI meme text generation (Groq llama-3.3-70b)
  5. Validation gate (confidence >= 2)
  6. Pillow text overlay (Impact-style top/bottom text)
  7. Anti-detection tweaks (slight crop, brightness shift)
  8. Upload to Catbox
  9. Callback to n8n webhook

Much lighter than process_video.py — no FFmpeg/Whisper needed.
Designed to run on GitHub Actions (Ubuntu, CPU only).
"""

import os
import sys
import json
import requests
import time
import random
import traceback
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")

GROQ_MODEL = "llama-3.3-70b-versatile"           # Primary text gen (best quality)
GROQ_TEXT_FALLBACK = "llama-3.1-8b-instant"       # Fallback text gen (separate rate limit, higher RPM)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision

MIN_CONFIDENCE = 2
WATERMARK_TEXT = "@Meme_Facteory"

WORK_DIR = "work"
OUTPUT_DIR = "output"

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    """Main pipeline orchestrator."""
    image_url = os.environ.get("IMAGE_URL")
    image_id = os.environ.get("IMAGE_ID")
    reddit_title = os.environ.get("REDDIT_TITLE", "")
    reddit_sub = os.environ.get("REDDIT_SUB", "")

    if not image_url or not image_id:
        print("ERROR: IMAGE_URL and IMAGE_ID are required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "image_id": image_id,
        "content_type": "image",
        "title": reddit_title,
        "status": "processing",
        "stages_completed": [],
        "stages_failed": [],
    }

    try:
        # ── Stage 1: Download Image ──────────────────────────────────────
        log_stage("DOWNLOAD IMAGE")
        raw_image = download_image(image_url, image_id)
        output["stages_completed"].append("download")

        # ── Stage 2: Validate Image ──────────────────────────────────────
        log_stage("VALIDATE IMAGE")
        validate_image(raw_image)
        output["stages_completed"].append("validate")

        # ── Stage 3: Groq Vision Captioning ───────────────────────────────
        log_stage("GROQ VISION CAPTIONING")
        vision_description = caption_image(raw_image)
        print(f"  Caption: {vision_description[:100]}")
        output["stages_completed"].append("groq_vision")

        # ── Stage 4: Context Builder ─────────────────────────────────────
        log_stage("CONTEXT BUILDER")
        context = build_context(reddit_title, reddit_sub, vision_description)
        print(f"  Emotion: {context['detected_emotion']}")
        print(f"  Scene: {context['scene_type']}")
        output["stages_completed"].append("context_build")

        # ── Stage 5: AI Meme Text Generation ─────────────────────────────
        log_stage("AI MEME GENERATION")
        meme_data = generate_meme_text(context)
        print(f"  Top text: {meme_data.get('top_text', 'N/A')}")
        print(f"  Bottom text: {meme_data.get('bottom_text', 'N/A')}")
        print(f"  Style: {meme_data.get('meme_style', 'N/A')}")
        print(f"  Confidence: {meme_data.get('confidence_score', 'N/A')}")
        output["stages_completed"].append("ai_generate")

        # ── Stage 6: Validation Gate ─────────────────────────────────────
        log_stage("VALIDATION")
        is_valid, reject_reasons = validate_meme(meme_data, context)

        if not is_valid:
            output["status"] = "skipped"
            output["reason"] = "; ".join(reject_reasons)
            print(f"  REJECTED: {output['reason']}")
            save_output(output, image_id)
            notify_n8n(output)
            return

        print("  PASSED ✅")
        output["stages_completed"].append("validation")

        # ── Stage 7: Pillow Text Overlay ─────────────────────────────────
        log_stage("RENDER IMAGE MEME")
        final_image = render_meme_image(raw_image, meme_data, image_id)
        output["stages_completed"].append("render")

        file_size = os.path.getsize(final_image) / 1024
        print(f"  Final: {file_size:.1f}KB")

        # ── Stage 8: Upload ──────────────────────────────────────────────
        log_stage("UPLOAD")
        download_url = upload_to_catbox(final_image)
        print(f"  URL: {download_url}")
        output["stages_completed"].append("upload")

        # ── Final Output ─────────────────────────────────────────────────
        output.update({
            "status": "ready",
            "download_url": download_url,
            "caption": meme_data.get("caption", ""),
            "meme_data": meme_data,
            "file_size_kb": round(file_size, 2),
        })

    except Exception as e:
        output["status"] = "failed"
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"\n❌ PIPELINE FAILED: {e}", file=sys.stderr)
        traceback.print_exc()

    # Always save output and notify
    save_output(output, image_id)
    notify_n8n(output)

    if output["status"] == "ready":
        print(f"\n✅ SUCCESS: Meme ready at {output.get('download_url')}")
    elif output["status"] == "skipped":
        print(f"\n⏭️ SKIPPED: {output.get('reason')}")
    else:
        print(f"\n❌ FAILED: {output.get('error')}")
        sys.exit(1)


# ─── Stage Functions ─────────────────────────────────────────────────────────

def download_image(url, image_id):
    """Stage 1: Download image from Reddit."""
    # Determine file extension from URL
    ext = ".jpg"
    for e in [".png", ".gif", ".webp", ".jpeg"]:
        if e in url.lower():
            ext = e
            break

    output_path = f"{WORK_DIR}/{image_id}_raw{ext}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*",
        "Referer": "https://www.reddit.com/",
    }

    # Clean up preview.redd.it URLs
    clean_url = url.replace("&amp;", "&")

    print(f"  Downloading: {clean_url[:80]}...")
    resp = requests.get(clean_url, headers=headers, stream=True, timeout=60, allow_redirects=True)

    if resp.status_code != 200:
        raise RuntimeError(f"Image download failed: HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise RuntimeError("Got HTML instead of image (Reddit blocked)")

    total_size = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total_size += len(chunk)

    if total_size < 5000:
        os.remove(output_path)
        raise RuntimeError(f"Downloaded file too small ({total_size} bytes)")

    print(f"  Downloaded: {output_path} ({total_size / 1024:.1f}KB)")
    return output_path


def validate_image(image_path):
    """Stage 2: Validate the downloaded image is usable."""
    from PIL import Image

    img = Image.open(image_path)
    width, height = img.size
    print(f"  Size: {width}x{height}, Format: {img.format}, Mode: {img.mode}")

    # Check minimum dimensions
    if width < 100 or height < 100:
        raise ValueError(f"Image too small ({width}x{height})")

    # Check for extremely large images
    if width > 8000 or height > 8000:
        print(f"  Resizing from {width}x{height}...")
        img.thumbnail((4000, 4000), Image.LANCZOS)
        img.save(image_path)
        print(f"  Resized to {img.size}")

    img.close()


def caption_image(image_path):
    """Stage 3: Describe image using Groq Vision API.
    
    Uses Groq llama-4-scout-17b-16e-instruct for fast, accurate image analysis.
    """
    import base64
    import io
    from PIL import Image

    if not GROQ_API_KEY:
        raise RuntimeError("Need GROQ_API_KEY for vision captioning")

    # Convert image to base64 (resize if too large to save tokens)
    img = Image.open(image_path).convert("RGB")
    max_dim = 512
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    vision_prompt = (
        "Describe this image in detail (3-5 sentences). Include:\n"
        "1. What is shown — people, objects, scene, setting\n"
        "2. Any TEXT visible in the image — read and quote it EXACTLY (tweets, messages, captions, signs, etc.)\n"
        "3. Expressions, emotions, and reactions visible\n"
        "4. If it's a screenshot of a conversation/tweet/post, transcribe the FULL text content\n"
        "Be specific and complete — capture everything that would be needed to understand the humor or context."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                }
            ]
        }
    ]

    # ── Direct Groq Vision call (only reliable provider) ────────────────
    if not GROQ_API_KEY:
        print("  No GROQ_API_KEY set — cannot analyze image")
        return "Could not analyze image — using reddit title for context"

    print(f"  Sending to Groq Vision ({GROQ_VISION_MODEL.split('/')[-1]})...")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )

        if response.status_code == 429:
            wait = min(int(response.headers.get("retry-after", 15)), 30)
            print(f"  Groq rate limited, waiting {wait}s...")
            time.sleep(wait)
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=30
            )

        if response.status_code != 200:
            print(f"  Groq returned {response.status_code}: {response.text[:150]}")
            return "Could not analyze image — using reddit title for context"

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            print(f"  Groq returned empty content")
            return "Could not analyze image — using reddit title for context"

        caption = content.strip()
        print(f"  [Groq] ✓ {caption}")
        print(f"  Caption length: {len(caption)} chars")
        return caption

    except Exception as e:
        print(f"  Groq vision failed: {e}")
        return "Could not analyze image — using reddit title for context"



# ─── Context Analysis ────────────────────────────────────────────────────────

def detect_emotion(vision_description, reddit_title):
    """Detect dominant emotion from image context."""
    text = (vision_description + " " + reddit_title).lower()

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


def classify_scene(vision_description, reddit_title):
    """Classify scene type from context."""
    text = (vision_description + " " + reddit_title).lower()

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


def build_context(reddit_title, reddit_sub, vision_description):
    """Stage 4: Build unified context from image analysis."""
    emotion = detect_emotion(vision_description, reddit_title)
    scene_type = classify_scene(vision_description, reddit_title)

    return {
        "reddit_title": reddit_title,
        "reddit_sub": reddit_sub,
        "vision_description": vision_description,
        "detected_emotion": emotion,
        "scene_type": scene_type,
    }


# ─── AI Meme Generation ─────────────────────────────────────────────────────

def generate_meme_text(context):
    """Stage 5: Generate meme text using Groq AI."""
    if not GROQ_API_KEY:
        raise RuntimeError("Need GROQ_API_KEY for text generation")

    prompt = f"""You are India's #1 viral meme creator for Gen-Z Instagram.
You make Hinglish (Hindi + English mix) image memes that get MILLIONS of likes.

═══ IMAGE CONTEXT ═══

Reddit Title: {context['reddit_title']}
Subreddit: {context['reddit_sub']}

Visual Description (Vision AI):
"{context['vision_description']}"

Detected Emotion: {context['detected_emotion']}
Scene Type: {context['scene_type']}

═══ RULES ═══

1. This is an IMAGE meme, not a video — text overlay style (top/bottom Impact text)
2. top_text MUST be 4-8 words in Hinglish — directly describe what's happening in the image
3. bottom_text is the punchline — 3-6 words, savage/funny twist
4. Together they should create a classic meme format with setup → punchline
5. Must be genuinely FUNNY, savage, cringe, or relatable — viral-worthy
6. The meme should feel like something @IndianDankMemes or @theindianmemer would post
7. Use POV/relatable/cringe/savage tone as appropriate
8. Understand the image first from the Vision AI description, then make a meme about THAT situation
9. Examples of good top/bottom combos:
   - "Jab dost bole 'bas 5 min'" / "2 ghante baad bhi wahi"
   - "POV: Tu exam hall mein" / "Aur sab likh rahe hain"
   - "When crush says 'you're like a brother'" / "*Rakhi bandh diya*"

═══ RETURN FORMAT ═══

Return ONLY valid JSON, no markdown, no explanation:
{{
  "situation_summary": "Brief 1-2 line description of what the image shows",
  "top_text": "4-8 word Hinglish setup text",
  "bottom_text": "3-6 word Hinglish punchline/twist",
  "meme_style": "top_bottom",
  "hashtags": ["10", "relevant", "indian", "meme", "hashtags"],
  "emotion_detected": "{context['detected_emotion']}",
  "confidence_score": 8,
  "caption": "Full Instagram caption — Hinglish hook + emojis + CTA + line break + hashtags"
}}"""

    system_msg = (
        "You are India's top meme page admin. You create viral Hinglish image memes "
        "that get millions of likes. You understand Indian culture, Gen-Z humor, "
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
        "max_tokens": 500,
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

        required = ["top_text", "confidence_score", "caption"]
        for field in required:
            if field not in meme_data:
                raise ValueError(f"Missing required field: {field}")

        if "bottom_text" not in meme_data:
            meme_data["bottom_text"] = ""

        print(f"  [{provider}] ✓ Meme generated")
        return meme_data

    except json.JSONDecodeError as e:
        print(f"  {provider} JSON parse failed: {e}")
    except Exception as e:
        print(f"  {provider} failed: {e}")

    return None


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_meme(meme_data, context):
    """Stage 6: Validate meme quality before rendering.

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

    # Check if vision is useless
    desc = context.get("vision_description", "").lower()
    if any(x in desc for x in ["unclear", "blurry", "dark", "nothing", "error"]):
        reasons.append("Vision description is unclear/unusable")

    # Check top_text length
    if len(meme_data.get("top_text", "")) > 120:
        reasons.append("top_text too long (> 120 chars)")

    return (len(reasons) == 0, reasons)


# ─── Image Rendering ────────────────────────────────────────────────────────

def render_meme_image(image_path, meme_data, image_id):
    """Stage 7: Render meme text overlay using Pillow.

    Classic Impact-style meme: bold text at top and bottom with black outline.
    Also applies anti-detection tweaks (slight crop + brightness shift).
    """
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

    top_text = meme_data.get("top_text", "").upper()
    bottom_text = meme_data.get("bottom_text", "").upper()

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # ── Scale to Instagram-friendly size ──────────────────────────────────
    # Target: 1080px wide, maintain aspect ratio (max 1350px tall for 4:5)
    target_width = 1080
    scale = target_width / width
    new_height = int(height * scale)
    # Cap height at 1350 (Instagram 4:5 max) or 1920 (9:16 stories)
    max_height = 1350
    if new_height > max_height:
        new_height = max_height

    img = img.resize((target_width, new_height), Image.LANCZOS)
    width, height = img.size

    # ── Anti-detection tweaks ─────────────────────────────────────────────
    # Slight random crop (1-3% from each edge)
    crop_px = int(min(width, height) * random.uniform(0.01, 0.03))
    img = img.crop((crop_px, crop_px, width - crop_px, height - crop_px))
    img = img.resize((width, height), Image.LANCZOS)

    # Subtle brightness/contrast shift
    brightness_factor = random.uniform(1.02, 1.08)
    contrast_factor = random.uniform(1.01, 1.05)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    draw = ImageDraw.Draw(img)

    # ── Load font ─────────────────────────────────────────────────────────
    font = _load_meme_font(width)

    # ── Draw top text ─────────────────────────────────────────────────────
    if top_text:
        _draw_outlined_text(draw, top_text, font, width, height, position="top")

    # ── Draw bottom text ──────────────────────────────────────────────────
    if bottom_text:
        _draw_outlined_text(draw, bottom_text, font, width, height, position="bottom")

    # ── Watermark ─────────────────────────────────────────────────────────
    wm_font_size = max(16, width // 50)
    try:
        wm_font = ImageFont.truetype(
            _find_font_path(),
            wm_font_size
        )
    except:
        wm_font = ImageFont.load_default()

    wm_bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=wm_font)
    wm_w = wm_bbox[2] - wm_bbox[0]
    wm_x = (width - wm_w) // 2
    wm_y = height - wm_font_size - 15
    # Semi-transparent watermark
    draw.text((wm_x + 1, wm_y + 1), WATERMARK_TEXT, font=wm_font, fill=(0, 0, 0, 180))
    draw.text((wm_x, wm_y), WATERMARK_TEXT, font=wm_font, fill=(255, 255, 255, 200))

    # ── Save ──────────────────────────────────────────────────────────────
    final_path = f"{WORK_DIR}/{image_id}_final.jpg"
    img.save(final_path, "JPEG", quality=92, optimize=True)
    print(f"  Saved: {final_path}")

    return final_path


def _draw_outlined_text(draw, text, font, img_width, img_height, position="top"):
    """Draw Impact-style outlined text centered at top or bottom.
    
    Uses pixel-based wrapping to ensure text never overflows the image edges.
    Auto-shrinks font if text still doesn't fit.
    """
    from PIL import ImageFont
    
    padding = int(img_width * 0.05)  # 5% padding on each side
    max_text_width = img_width - (padding * 2)
    
    # Pixel-based word wrap
    lines = _pixel_word_wrap(draw, text, font, max_text_width)
    joined = "\n".join(lines)
    
    # Check if text fits — if too many lines, shrink font
    text_bbox = draw.multiline_textbbox((0, 0), joined, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    max_text_height = int(img_height * 0.35)  # Max 35% of image height per text block
    
    # Auto-shrink if text is too tall
    current_font = font
    shrink_attempts = 0
    while text_h > max_text_height and shrink_attempts < 5:
        shrink_attempts += 1
        new_size = max(18, current_font.size - 8)
        try:
            font_path = _find_font_path()
            current_font = ImageFont.truetype(font_path, new_size) if font_path else ImageFont.load_default()
        except:
            break
        lines = _pixel_word_wrap(draw, text, current_font, max_text_width)
        joined = "\n".join(lines)
        text_bbox = draw.multiline_textbbox((0, 0), joined, font=current_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    
    x = max(padding, (img_width - text_w) // 2)

    if position == "top":
        y = max(10, int(img_height * 0.02))
        # Ensure top text doesn't go past 40% of image
        if y + text_h > int(img_height * 0.40):
            y = max(10, int(img_height * 0.40) - text_h)
    else:
        y = img_height - text_h - max(20, int(img_height * 0.03))
        # Ensure bottom text doesn't go above middle of image
        y = max(int(img_height * 0.55), y)
        # Ensure bottom text doesn't go below image
        if y + text_h > img_height - 10:
            y = img_height - text_h - 10

    # Final safety clamp — always within bounds
    y = max(5, min(y, img_height - text_h - 5))

    # Draw black outline (stroke)
    outline_width = max(3, img_width // 250)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.multiline_text(
                (x + dx, y + dy), joined, font=current_font,
                fill="black", align="center"
            )

    # Draw white text on top
    draw.multiline_text((x, y), joined, font=current_font, fill="white", align="center")


def _word_wrap(text, max_chars):
    """Wrap text into lines of max_chars width (character-based fallback)."""
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

    return lines if lines else [text]


def _pixel_word_wrap(draw, text, font, max_width):
    """Wrap text based on actual pixel width using the given font.
    
    This ensures text never overflows the image edges regardless of
    font size or character width.
    """
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test_line = current + " " + word if current else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        
        if line_width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test_line

    if current:
        lines.append(current)

    return lines if lines else [text]


def _load_meme_font(img_width):
    """Load the best available bold font, sized proportionally to image width."""
    from PIL import ImageFont

    font_size = max(28, img_width // 16)  # ~67px for 1080px wide image

    font_path = _find_font_path()
    try:
        return ImageFont.truetype(font_path, font_size)
    except:
        pass

    # Fallback: try system fonts
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except:
                continue

    print("  WARNING: No TrueType font found, using default")
    return ImageFont.load_default()


def _find_font_path():
    """Find the best available font (Noto Sans preferred for Hindi support)."""
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
    return ""


# ─── Upload ──────────────────────────────────────────────────────────────────

def upload_to_catbox(image_path):
    """Stage 8: Upload to temporary file hosting."""
    file_size = os.path.getsize(image_path) / 1024
    print(f"  Uploading {file_size:.1f}KB...")

    # Try litterbox first (72h temp hosting)
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload", "time": "72h"},
                timeout=60,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Litterbox failed: {e}")

    # Fallback to permanent catbox
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                files={"fileToUpload": f},
                data={"reqtype": "fileupload"},
                timeout=60,
            )
        url = response.text.strip()
        if url.startswith("https://"):
            return url
    except Exception as e:
        print(f"  Catbox failed: {e}")

    raise RuntimeError("All upload methods failed")


# ─── Utilities ───────────────────────────────────────────────────────────────

def log_stage(name):
    """Print a visible stage header."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")


def save_output(data, image_id):
    """Save structured output JSON for debugging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{image_id}.json"
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
    print("║      ORIGINAL MEME FACTORY — Image Processing Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Vision: Groq {GROQ_VISION_MODEL.split('/')[-1]}")
    print(f"  Text gen: Groq {GROQ_MODEL} (fallback: {GROQ_TEXT_FALLBACK})")
    print(f"  Min confidence: {MIN_CONFIDENCE}")

    main()

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
