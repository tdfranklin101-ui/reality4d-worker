
TC-S Network — Reality 4D RunPod Serverless Worker
What This Is
A custom RunPod Serverless worker that runs LTX-2 Pro video generation with full character identity pipeline support. Accepts the exact payload format from Prompt a Movie — identity_blocks, scene_context, render_config — without flattening anything into generic prompts.

Prerequisites
RunPod account (runpod.io)
Docker installed locally (for building the image)
Docker Hub account (or any container registry)
Step 1: Build the Docker Image
cd runpod-worker
docker build --platform linux/amd64 -t YOUR_DOCKERHUB/reality4d-worker:v1.0 .
docker push YOUR_DOCKERHUB/reality4d-worker:v1.0

Note: The image does NOT include model weights. On first cold start, the worker downloads LTX-Video (13GB) from HuggingFace automatically. Subsequent runs reuse the cached model. To eliminate cold-start downloads, attach a RunPod Network Volume ($0.91/month) mounted at /models.

Build time: ~10-15 minutes (installs PyTorch, diffusers, rembg, etc.).

Step 2: Create RunPod Serverless Endpoint
Log in to runpod.io
Go to Serverless in the left sidebar
Click New Endpoint
Click Import from Docker Registry
Enter your image: YOUR_DOCKERHUB/reality4d-worker:v1.0
Configure the endpoint:
Setting	Recommended Value
Endpoint Name	reality4d-ltx2pro
GPU Type	A100 40GB or L40S 48GB
Min Active Workers	0 (scales to zero)
Max Workers	3
Idle Timeout	30 seconds
Execution Timeout	600 seconds (10 min)
Flash Boot	Enabled
(Optional) Environment Variables for S3 upload:
Variable	Value
UPLOAD_METHOD	s3
S3_BUCKET	your-bucket-name
S3_REGION	us-east-1
S3_PREFIX	renders/
AWS_ACCESS_KEY_ID	your-key
AWS_SECRET_ACCESS_KEY	your-secret
If you skip S3 config, videos are returned as base64 in the response body.

Click Create Endpoint
Copy the Endpoint ID — this goes into Prompt a Movie
Step 3: Update Prompt a Movie Secrets
In Replit, update these environment secrets:

RUNPOD_API_KEY — your RunPod API key (from runpod.io → Settings → API Keys)
RUNPOD_ENDPOINT_ID — the endpoint ID from step 2
Step 4: Test
Test locally before deploying:

cd runpod-worker
python src/rp_handler.py --test_input src/test_input.json

Test on RunPod after deployment:

curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @src/test_input.json

GPU Selection Guide
GPU	VRAM	$/hr	Best For
RTX 4090	24GB	~$0.44	Short clips, distilled model
L40S	48GB	~$0.89	Good balance of cost & quality
A100 40GB	40GB	~$1.14	Full quality, reliable
A100 80GB	80GB	~$1.64	Longer clips, higher resolution
H100 SXM	80GB	~$2.71	Maximum quality & speed
Cost per 5-second clip (A100 40GB): ~$0.02-0.03

How It Works
Prompt a Movie sends a Reality4DJobInput payload to RunPod
The worker receives it in rp_handler.py
build_prompt() assembles a full cinematic prompt from:
identity_blocks (character visual signatures, motion directives)
scene_context (description, mood, dialogue cues, story beats)
style_prefix and negative_prompt
LTX-2 Pro generates the video frames
If composite_mode is "depth_aware" (World Labs compositing):
rembg AI strips the background from every frame (u2net_human_seg model)
Frames are exported as WebM with alpha transparency (VP9, yuva420p)
Prompt a Movie composites the transparent character onto a World Labs 3D environment
If composite_mode is "direct" (standard render):
Frames are exported as MP4 via imageio/ffmpeg
Video is either uploaded to S3 or returned as base64
Prompt a Movie receives the output and stores the clip
Payload Format
See src/test_input.json for a complete example of the input payload.

The worker returns:
{
  "video_url": "https://...",
  "duration_actual": 5.0,
  "composite_mode": "direct",
  "clip_id": "...",
  "project_id": "...",
  "generation_time_seconds": 85.2,
  "resolution": "768x512",
  "num_frames": 121,
  "fps": 24,
  "identity_block_count": 1,
  "metadata": { ... }
}
The worker returns:
