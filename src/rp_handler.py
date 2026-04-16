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
        existing_files = os.listdir(MODEL_DIR)
        total_size = sum(
            os.path.getsize(os.path.join(MODEL_DIR, f))
            for f in existing_files
            if os.path.isfile(os.path.join(MODEL_DIR, f))
        )
        print(f"[Worker] Model already present at {MODEL_DIR} ({len(existing_files)} files, {total_size / 1e9:.2f} GB)")
        return

    print(f"[Worker] Model not found at {MODEL_DIR} — downloading Lightricks/LTX-Video from HuggingFace...")
    print(f"[Worker] This is a ~13 GB download and may take 5-15 minutes on first cold start.")
    dl_start = time.time()

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            "Lightricks/LTX-Video",
            local_dir=MODEL_DIR,
            ignore_patterns=["*.md", "*.txt", "LICENSE*", "*.png", "*.jpg"],
        )
    except Exception as e:
        elapsed = time.time() - dl_start
        partial_files = os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []
        print(f"[Worker] ERROR: Model download failed after {elapsed:.1f}s: {e}")
        print(f"[Worker] Partial files in {MODEL_DIR}: {partial_files}")
        raise

    elapsed = time.time() - dl_start
    dl_files = os.listdir(MODEL_DIR)
    total_size = sum(
        os.path.getsize(os.path.join(MODEL_DIR, f))
        for f in dl_files
        if os.path.isfile(os.path.join(MODEL_DIR, f))
    )
    print(f"[Worker] Model downloaded in {elapsed:.1f}s ({len(dl_files)} files, {total_size / 1e9:.2f} GB)")


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


def build_prompt(inp, composite_mode="direct"):
    """
    Assemble a full cinematic prompt from structured input fields.
    Preserves every character identity detail, scene context, and style directive.

    When composite_mode is 'depth_aware', renders a photorealistic character
    performance against a simple neutral backdrop. AI background removal then
    strips the background entirely, producing transparent frames. World Labs
    3D environments fill in behind the character — no chroma key needed.
    """
    parts = []
    is_character_pass = composite_mode in ("depth_aware", "green_screen")

    if is_character_pass:
        parts.append(
            "Photorealistic cinematic footage of a real human actor performing "
            "against a plain neutral backdrop. Soft even studio lighting on the "
            "actor. Clean separation between the actor and the background."
        )

    style_prefix = inp.get("scene_context", {}).get("style_prefix", "")
    if style_prefix:
        parts.append(style_prefix)

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
        if is_character_pass:
            parts.append(f"Action/performance: {base_prompt}")
        else:
            parts.append(base_prompt)

    scene_desc = inp.get("scene_context", {}).get("scene_description", "")
    if not is_character_pass and scene_desc and scene_desc != base_prompt:
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


def build_negative_prompt(inp, composite_mode="direct"):
    """Build negative prompt from scene_context or use default."""
    custom = inp.get("scene_context", {}).get("negative_prompt", "")
    default = (
        "3D character render, cartoon, anime, CGI, plastic skin, doll-like, "
        "mannequin, video game, low quality, blurry, distorted, watermark, "
        "text overlay, gibberish glyphs, faux runes"
    )
    is_character_pass = composite_mode in ("depth_aware", "green_screen")
    if is_character_pass:
        default += (
            ", cluttered background, detailed environment, busy scenery"
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


rembg_session = None

def remove_background_from_frames(frames):
    """
    Strip background from each frame using rembg AI matting.
    Returns RGBA PIL Images with transparent background — no chroma key needed.
    The AI model understands human boundaries including hair, translucent fabric,
    and fine edges far better than any color-based keying.
    """
    global rembg_session
    from rembg import remove, new_session
    from PIL import Image
    import numpy as np

    if rembg_session is None:
        print("[Worker] Loading rembg background removal model (u2net_human_seg)...")
        rembg_session = new_session("u2net_human_seg")
        print("[Worker] rembg model loaded")

    transparent_frames = []
    for i, frame in enumerate(frames):
        if isinstance(frame, np.ndarray):
            frame_img = Image.fromarray(frame)
        else:
            frame_img = frame

        rgba = remove(frame_img, session=rembg_session, post_process_mask=True)
        transparent_frames.append(rgba)

        if i == 0 or (i + 1) % 24 == 0:
            print(f"[Worker] Background removed: frame {i + 1}/{len(frames)}")

    print(f"[Worker] Background removal complete: {len(transparent_frames)} transparent frames")
    return transparent_frames


def export_video(frames, fps, clip_id, is_character_pass=False):
    """Export frames to video. Character passes export as WebM with alpha transparency.
    Full scenes export as MP4."""
    import imageio
    import numpy as np

    if is_character_pass:
        transparent_frames = remove_background_from_frames(frames)

        tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir="/tmp")
        tmp_path = tmp.name
        tmp.close()

        rgba_arrays = []
        for f in transparent_frames:
            rgba_arrays.append(np.array(f.convert("RGBA")))

        writer = imageio.get_writer(tmp_path, fps=fps, codec="libvpx-vp9",
                                     output_params=["-pix_fmt", "yuva420p", "-auto-alt-ref", "0"])
        for arr in rgba_arrays:
            writer.append_data(arr)
        writer.close()

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"[Worker] Exported transparent character video: {tmp_path} ({file_size_mb:.1f} MB)")

        if UPLOAD_METHOD == "s3" and S3_BUCKET:
            key = f"{clip_id}_{int(time.time())}_alpha.webm"
            url = upload_to_s3(tmp_path, key)
            os.unlink(tmp_path)
            return {"video_url": url, "method": "s3", "format": "webm_alpha"}

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(tmp_path)
        return {"video_b64": video_b64, "method": "base64", "format": "webm_alpha"}

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    tmp_path = tmp.name
    tmp.close()

    imageio.mimwrite(tmp_path, frames, fps=fps, quality=8)

    if UPLOAD_METHOD == "s3" and S3_BUCKET:
        key = f"{clip_id}_{int(time.time())}.mp4"
        url = upload_to_s3(tmp_path, key)
        os.unlink(tmp_path)
        return {"video_url": url, "method": "s3", "format": "mp4"}

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return {"video_b64": video_b64, "method": "base64", "format": "mp4"}


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

    prompt = build_prompt(inp, composite_mode)
    negative_prompt = build_negative_prompt(inp, composite_mode)

    is_character_pass = composite_mode in ("depth_aware", "green_screen")
    render_mode_label = "CHARACTER PASS (AI bg removal → transparent)" if is_character_pass else "FULL SCENE"
    print(f"[Worker] Render mode: {render_mode_label}, composite_mode={composite_mode}")
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
        result = export_video(frames, fps, clip_id, is_character_pass=is_character_pass)
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
        "video_format": result.get("format", "mp4"),
        "has_alpha": is_character_pass,
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
            "render_pass": "character_isolated" if is_character_pass else "full_scene",
        },
    }

    print(f"[Worker] Job complete: clip={clip_id}, duration={duration_actual:.2f}s, gen_time={gen_time:.1f}s")
    return output_payload


runpod.serverless.start({"handler": handler})

