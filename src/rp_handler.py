"""
TC-S Network — Reality 4D RunPod Serverless Worker
Custom handler for LTX-2 Pro video generation with full character identity pipeline.

Accepts the Reality4DJobInput payload directly from Prompt a Movie,
preserving identity_blocks, scene_context, world_config, and render_config
without flattening into generic prompts.
"""

import runpod
import torch
import os
import time
import tempfile
import base64
import json
import traceback

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/ltx-video")
UPLOAD_METHOD = os.environ.get("UPLOAD_METHOD", "base64")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", "renders/")

pipe = None


def ensure_model_downloaded():
    """Download LTX-Video model if not already present."""
    marker = os.path.join(MODEL_DIR, "model_index.json")
    if os.path.exists(marker):
        print(f"[Worker] Model already present at {MODEL_DIR}")
        return

    print(f"[Worker] Model not found — downloading Lightricks/LTX-Video...")
    dl_start = time.time()

    from huggingface_hub import snapshot_download
    snapshot_download(
        "Lightricks/LTX-Video",
        local_dir=MODEL_DIR,
        ignore_patterns=["*.md", "*.txt", "LICENSE*"],
    )

    elapsed = time.time() - dl_start
    print(f"[Worker] Model downloaded in {elapsed:.1f}s")


def load_model():
    """Load LTX-Video pipeline once at worker startup."""
    global pipe
    if pipe is not None:
        return

    ensure_model_downloaded()

    print(f"[Worker] Loading LTX-Video model from {MODEL_DIR}")
    start = time.time()

    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.enable_model_cpu_offload()

    elapsed = time.time() - start
    print(f"[Worker] Model loaded in {elapsed:.1f}s")


def build_prompt(inp):
    """
    Assemble a full cinematic prompt from structured input fields.
    Preserves every character identity detail, scene context, and style directive.
    """
    parts = []

    style_prefix = inp.get("scene_context", {}).get("style_prefix", "")
    if style_prefix:
        parts.append(style_prefix)

    scene_desc = inp.get("scene_context", {}).get("scene_description", "")
    mood = inp.get("scene_context", {}).get("mood", "cinematic")
    scene_heading = inp.get("scene_context", {}).get("scene_heading", "")

    if scene_heading:
        parts.append(f"{scene_heading}.")

    identity_blocks = inp.get("identity_blocks", [])
    for block in identity_blocks:
        name = block.get("name", "")
        identity = block.get("identity", "")
        motion = block.get("motion", "")

        char_desc = f"{name}: {identity}"
        if motion:
            char_desc += f" {motion}"
        parts.append(char_desc)

    base_prompt = inp.get("prompt", "")
    if base_prompt:
        parts.append(base_prompt)

    if scene_desc and scene_desc != base_prompt:
        parts.append(scene_desc)

    dialogue_cue = inp.get("scene_context", {}).get("dialogue_cue", "")
    if dialogue_cue:
        parts.append(f"Speaking: \"{dialogue_cue}\"")

    story_beat = inp.get("scene_context", {}).get("story_beat", "")
    if story_beat:
        parts.append(f"Story continuity: {story_beat}")

    parts.append(f"Mood: {mood}.")

    full_prompt = " ".join(p.strip() for p in parts if p.strip())

    return full_prompt


def build_negative_prompt(inp):
    """Build negative prompt from scene_context or use default."""
    custom = inp.get("scene_context", {}).get("negative_prompt", "")
    default = (
        "3D character render, cartoon, anime, CGI, plastic skin, doll-like, "
        "mannequin, video game, low quality, blurry, distorted, watermark, "
        "text overlay, gibberish glyphs, faux runes"
    )
    if custom:
        return f"{custom}, {default}"
    return default


def parse_resolution(res_str):
    """Parse resolution string like '1920x1080' into (width, height)."""
    if not res_str or "x" not in str(res_str):
        return 768, 512

    try:
        w, h = str(res_str).split("x")
        w, h = int(w), int(h)
        w = max(256, min(1920, (w // 32) * 32))
        h = max(256, min(1080, (h // 32) * 32))
        return w, h
    except (ValueError, TypeError):
        return 768, 512


def duration_to_frames(duration_seconds, fps=24):
    """Convert duration in seconds to number of frames.
    LTX requires num_frames = k*8 + 1 for some integer k."""
    raw_frames = int(duration_seconds * fps)
    k = max(1, raw_frames // 8)
    num_frames = k * 8 + 1
    return min(num_frames, 257)


def upload_to_s3(file_path, key):
    """Upload a file to S3 and return the public URL."""
    import boto3

    s3 = boto3.client("s3", region_name=S3_REGION)
    full_key = f"{S3_PREFIX}{key}"

    s3.upload_file(
        file_path,
        S3_BUCKET,
        full_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{full_key}"
    return url


def export_video(frames, fps, clip_id):
    """Export PIL frames to MP4 and return file path or URL."""
    import imageio

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    tmp_path = tmp.name
    tmp.close()

    imageio.mimwrite(tmp_path, frames, fps=fps, quality=8)

    if UPLOAD_METHOD == "s3" and S3_BUCKET:
        key = f"{clip_id}_{int(time.time())}.mp4"
        url = upload_to_s3(tmp_path, key)
        os.unlink(tmp_path)
        return {"video_url": url, "method": "s3"}

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return {"video_b64": video_b64, "method": "base64"}


def handler(event):
    """
    RunPod serverless handler.
    Accepts Reality4DJobInput payload from Prompt a Movie.
    """
    try:
        load_model()
    except Exception as e:
        return {"error": f"Model load failed: {str(e)}", "traceback": traceback.format_exc()}

    inp = event.get("input", {})
    clip_id = inp.get("clip_id", f"clip_{int(time.time())}")
    project_id = inp.get("project_id", "unknown")

    print(f"[Worker] Job received: clip={clip_id}, project={project_id}")

    render_config = inp.get("render_config", {})
    duration_seconds = render_config.get("duration_seconds", 5)
    resolution = render_config.get("resolution", "768x512")
    fps = render_config.get("fps", 24)
    with_audio = render_config.get("with_audio", False)
    composite_mode = render_config.get("composite_mode", "direct")

    width, height = parse_resolution(resolution)
    num_frames = duration_to_frames(duration_seconds, fps)

    prompt = build_prompt(inp)
    negative_prompt = build_negative_prompt(inp)

    print(f"[Worker] Generating: {width}x{height}, {num_frames} frames ({duration_seconds}s @ {fps}fps)")
    print(f"[Worker] Prompt ({len(prompt)} chars): {prompt[:200]}...")
    print(f"[Worker] Negative: {negative_prompt[:100]}...")

    identity_summary = []
    for block in inp.get("identity_blocks", []):
        identity_summary.append({
            "character_id": block.get("character_id"),
            "name": block.get("name"),
            "identity_length": len(block.get("identity", "")),
            "motion": block.get("motion", "")[:80],
        })
    print(f"[Worker] Identity blocks: {json.dumps(identity_summary)}")

    gen_start = time.time()

    try:
        generator = torch.Generator("cuda").manual_seed(
            hash(clip_id) % (2**32)
        )

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=3.0,
            generator=generator,
        )

        frames = output.frames[0]
        gen_time = time.time() - gen_start
        print(f"[Worker] Generation complete: {len(frames)} frames in {gen_time:.1f}s")

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "error": f"GPU out of memory for {width}x{height} @ {num_frames} frames. Try lower resolution or shorter duration.",
            "clip_id": clip_id,
        }
    except Exception as e:
        return {
            "error": f"Generation failed: {str(e)}",
            "traceback": traceback.format_exc(),
            "clip_id": clip_id,
        }

    try:
        result = export_video(frames, fps, clip_id)
    except Exception as e:
        return {
            "error": f"Video export failed: {str(e)}",
            "traceback": traceback.format_exc(),
            "clip_id": clip_id,
        }

    total_time = time.time() - gen_start
    duration_actual = len(frames) / fps

    output_payload = {
        "video_url": result.get("video_url", ""),
        "video_b64": result.get("video_b64", ""),
        "duration_actual": round(duration_actual, 2),
        "composite_mode": composite_mode,
        "clip_id": clip_id,
        "project_id": project_id,
        "generation_time_seconds": round(gen_time, 2),
        "total_time_seconds": round(total_time, 2),
        "resolution": f"{width}x{height}",
        "num_frames": len(frames),
        "fps": fps,
        "prompt_length": len(prompt),
        "identity_block_count": len(inp.get("identity_blocks", [])),
        "metadata": {
            "content_mode": inp.get("content_mode", "live_action"),
            "production_type": inp.get("production_type", ""),
            "world_config_present": bool(inp.get("world_config", {}).get("world_id")),
            "branching_enabled": bool(inp.get("branching", {}).get("is_branch_point")),
            "upload_method": result.get("method", "unknown"),
        },
    }

    print(f"[Worker] Job complete: clip={clip_id}, duration={duration_actual:.2f}s, gen_time={gen_time:.1f}s")
    return output_payload


runpod.serverless.start({"handler": handler})
