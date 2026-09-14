"""
post_production_agent.py

The Strands agent for the Multi-Modal Post-Production Assistant. Runs
LOCALLY (it needs local ffmpeg, local file access, and the local CapCut
desktop app's drafts folder — none of which a remote server can reach).
"""

import base64
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
import whisper
import numpy as np
import boto3
import pycapcut as cc
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionError as BotoConnectionError
from strands import Agent, tool


from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
AWS_REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-pro-v1:0"

# --- Claude Vision Configuration ---
# --- Claude Vision Configuration ---
# Switched to Sonnet 4.6 because the subscription is already active
VISION_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

TRANSCRIBE_S3_BUCKET = "postprod-audio-temp-98765"

# Used exclusively for evaluate_video_frame scoring service
AGENTCORE_RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:049584025361:runtime/PostProdScoring_PostProdScoring-pe1ItWFmrd"
AGENTCORE_REGION = "us-east-1"

_BOTO_CLIENT_CONFIG = BotoConfig(connect_timeout=10, read_timeout=30)

MAX_CONCURRENT_BEDROCK_CALLS = 3
_bedrock_semaphore = threading.Semaphore(MAX_CONCURRENT_BEDROCK_CALLS)

THROTTLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "TooManyRequestsException",
}
RETRYABLE_CONNECTION_ERRORS = (EndpointConnectionError, BotoConnectionError)
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------
# REMOTE SCORING & CLAUDE VISION HELPERS
# ---------------------------------------------------------------------
_agentcore_client = None

def _get_agentcore_client():
    global _agentcore_client
    if _agentcore_client is None:
        _agentcore_client = boto3.client(
            "bedrock-agentcore", region_name=AGENTCORE_REGION, config=_BOTO_CLIENT_CONFIG
        )
    return _agentcore_client

def _invoke_remote_scoring(payload: dict) -> dict:
    if not AGENTCORE_RUNTIME_ARN:
        raise RuntimeError(
            "AGENTCORE_RUNTIME_ARN is not set — deploy agentcore_scoring_service.py "
            "first and paste the resulting agentRuntimeArn into this constant."
        )

    client = _get_agentcore_client()
    body = json.dumps(payload).encode("utf-8")
    session_id = f"post-prod-{uuid.uuid4()}"

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _bedrock_semaphore:
                response = client.invoke_agent_runtime(
                    agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
                    runtimeSessionId=session_id,
                    payload=body,
                )
            response_body = response["response"].read()
            return json.loads(response_body)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            last_err = e
            if error_code in THROTTLE_ERROR_CODES and attempt < MAX_RETRIES:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"    [agentcore] {error_code}, retry {attempt}/{MAX_RETRIES} after {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
        except RETRYABLE_CONNECTION_ERRORS as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"    [agentcore] {type(e).__name__}, retry {attempt}/{MAX_RETRIES} after {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
    raise last_err

def _invoke_claude_vision(images_b64: list[str], timestamps: list[float], content_query: str) -> dict:
    """Sends extracted video frames directly to Claude for high-resolution visual analysis."""
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    
    # 1. Build the multimodal payload for Claude
    content_blocks = []
    for i, b64_data in enumerate(images_b64):
        content_blocks.append({"text": f"Frame Index: {i} (Timestamp: {timestamps[i]}s)"})
        content_blocks.append({
            "image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64decode(b64_data)}
            }
        })
        
    content_blocks.append({
        "text": f"Analyze these sequential video frames. Find any frames that match this description: '{content_query}'. "
                f"You must return EXACTLY a JSON array containing the integer Frame Indexes that match (e.g., [2, 3]). "
                f"If no frames match, return an empty array []. Do not include markdown or explanations."
    })
    
    try:
        response = client.converse(
            modelId=VISION_MODEL_ID, 
            messages=[{"role": "user", "content": content_blocks}],
            inferenceConfig={"temperature": 0.0}
        )
        
        llm_text = response['output']['message']['content'][0]['text']
        
        # Safely extract the JSON array using regex to bypass conversational filler
        match = re.search(r'\[.*?\]', llm_text, re.DOTALL)
        if match:
            matches = json.loads(match.group(0))
            # Ensure the output is strictly a list of integers
            if isinstance(matches, list):
                return {"matches": [int(m) for m in matches if str(m).isdigit()]}
                
        return {"matches": []}
        
    except Exception as e:
        print(f"    [Claude Vision Error]: {str(e)}")
        return {"matches": []}

# --- Video Chunking Config ---
MAX_SAMPLE_FRAMES = 35
MIN_SAMPLE_INTERVAL_SEC = 0.5
LONG_VIDEO_THRESHOLD_SEC = 180.0
CHUNK_LENGTH_SEC = 45.0
CHUNK_OVERLAP_SEC = 5.0
MERGE_GAP_THRESHOLD_SEC = 1.0
MAX_CHUNK_WORKERS = 3

_REENCODE_CACHE_DIR = Path(tempfile.gettempdir()) / "post_production_agent_reencoded"

def _get_codec_info(video_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt",
                "-of", "default=noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    info = {}
    for line in (result.stdout or "").lower().splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            info[key.strip()] = val.strip()
    return info

def _incompatibility_reasons(info: dict) -> list[str]:
    codec_name = info.get("codec_name", "")
    pix_fmt = info.get("pix_fmt", "")
    is_hevc = codec_name in ("hevc", "h265")
    is_10bit = "10le" in pix_fmt or "10be" in pix_fmt or "p010" in pix_fmt

    reasons = []
    if is_hevc:
        reasons.append(f"codec is {codec_name.upper()} (HEVC/H.265)")
    if is_10bit:
        reasons.append(f"10-bit color ({pix_fmt})")
    return reasons

def _reencoded_cache_path(video_path: Path) -> Path:
    _REENCODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        mtime = int(video_path.stat().st_mtime)
    except OSError:
        mtime = 0
    key = hashlib.sha1(f"{video_path.resolve()}::{mtime}".encode("utf-8")).hexdigest()[:16]
    return _REENCODE_CACHE_DIR / f"{video_path.stem}_{key}_8bit.mp4"

def _ensure_capcut_compatible(video_path: Path) -> tuple[Path, str | None, str | None]:
    info = _get_codec_info(video_path)
    if not info:
        return video_path, None, None

    reasons = _incompatibility_reasons(info)
    if not reasons:
        return video_path, None, None

    reason_str = ", ".join(reasons)
    cache_path = _reencoded_cache_path(video_path)
    if cache_path.exists():
        note = f"{video_path.name}: auto-re-encoded ({reason_str}) — reused cached 8-bit copy"
        print(f"    [_ensure_capcut_compatible] {note}")
        return cache_path, note, None

    print(f"    [_ensure_capcut_compatible] {video_path.name}: {reason_str} — auto re-encoding...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-preset", "medium",
                "-c:a", "aac", "-b:a", "192k",
                str(cache_path),
            ],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
        err = f"{video_path.name} needs re-encoding ({reason_str}) but ffmpeg failed: {stderr[:300]}"
        print(f"    [_ensure_capcut_compatible] FAILED: {err}")
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        return video_path, None, err
    except FileNotFoundError:
        err = f"{video_path.name} needs re-encoding ({reason_str}) but ffmpeg is not on PATH"
        print(f"    [_ensure_capcut_compatible] FAILED: {err}")
        return video_path, None, err

    note = f"{video_path.name}: auto-re-encoded to 8-bit H.264 ({reason_str})"
    print(f"    [_ensure_capcut_compatible] {note}")
    return cache_path, note, None

def _get_video_duration(video_path: Path) -> float:
    # Try ffprobe first
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        # Fallback to ffmpeg if ffprobe is missing or fails
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(video_path)],
                capture_output=True, text=True
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass
        return 10.0 # Absolute fallback to prevent crashes

def _extract_frames(video_path: Path, out_dir: Path, count: int = 3) -> list[Path]:
    duration = _get_video_duration(video_path)
    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"frame_{i}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out_path),
            ],
            capture_output=True, check=True,
        )
        frame_paths.append(out_path)
    return frame_paths

def _encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _compute_sample_timestamps(duration: float, max_frames: int = MAX_SAMPLE_FRAMES,
                                min_interval: float = MIN_SAMPLE_INTERVAL_SEC) -> list[float]:
    if duration <= 0:
        return [0.0]
    safe_max = max(0.0, duration - max(0.15, duration * 0.01))
    max_by_interval = max(2, int(duration // min_interval) + 1)
    n = min(max_frames, max_by_interval)
    if n <= 1:
        return [0.0]
    raw_timestamps = [duration * i / (n - 1) for i in range(n)]
    return [min(t, safe_max) for t in raw_timestamps]

def _extract_frame_at(video_path: Path, timestamp: float, out_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out_path),
            ],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError:
        nudged_ts = max(0.0, timestamp - 0.5)
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(nudged_ts), "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out_path),
            ],
            capture_output=True, check=True,
        )

def _merge_matches_into_scenes(matches: list[int], timestamps: list[float], duration: float) -> list[dict]:
    valid_range = range(len(timestamps))
    raw_matches = sorted(set(matches))
    matches = [m for m in raw_matches if m in valid_range]
    if not matches:
        return []

    interval = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else duration
    half_pad = interval / 2

    runs = []
    run_start = matches[0]
    prev = matches[0]
    for idx in matches[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((run_start, prev))
        run_start = idx
        prev = idx
    runs.append((run_start, prev))

    scenes = []
    for start_idx, end_idx in runs:
        t_start = max(0.0, timestamps[start_idx] - half_pad)
        t_end = min(duration, timestamps[end_idx] + half_pad)
        scenes.append({
            "start_sec": round(t_start, 2),
            "end_sec": round(t_end, 2),
            "duration_sec": round(t_end - t_start, 2),
        })
    return scenes

def _scan_window(path: Path, content_query: str, window_start: float, window_end: float) -> list[dict]:
    window_duration = window_end - window_start
    local_timestamps = _compute_sample_timestamps(window_duration)
    global_timestamps = [window_start + t for t in local_timestamps]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_paths = []
        for i, ts in enumerate(global_timestamps):
            out_path = tmp_dir / f"frame_{i}.jpg"
            _extract_frame_at(path, ts, out_path)
            frame_paths.append(out_path)

        images_b64 = [_encode_image_b64(fp) for fp in frame_paths]

    parsed = _invoke_claude_vision(images_b64, global_timestamps, content_query)
    matches = parsed.get("matches", [])
    local_scenes = _merge_matches_into_scenes(matches, local_timestamps, window_duration)
    global_scenes = []
    for s in local_scenes:
        global_scenes.append({
            "start_sec": round(window_start + s["start_sec"], 2),
            "end_sec": round(window_start + s["end_sec"], 2),
            "duration_sec": s["duration_sec"],
        })
    return global_scenes

def _build_chunks(duration: float) -> list[tuple]:
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + CHUNK_LENGTH_SEC, duration)
        chunks.append((start, end))
        if end >= duration:
            break
        start = end - CHUNK_OVERLAP_SEC
    return chunks

def _merge_scenes_across_chunks(scenes: list[dict]) -> list[dict]:
    if not scenes:
        return []
    scenes = sorted(scenes, key=lambda s: s["start_sec"])
    merged = [dict(scenes[0])]
    for s in scenes[1:]:
        last = merged[-1]
        if s["start_sec"] - last["end_sec"] <= MERGE_GAP_THRESHOLD_SEC:
            last["end_sec"] = max(last["end_sec"], s["end_sec"])
            last["duration_sec"] = round(last["end_sec"] - last["start_sec"], 2)
        else:
            merged.append(dict(s))
    return merged

def _scan_long_video(path: Path, content_query: str, duration: float) -> list[dict]:
    chunks = _build_chunks(duration)
    all_scenes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CHUNK_WORKERS) as pool:
        futures = {
            pool.submit(_scan_window, path, content_query, c_start, c_end): (c_start, c_end)
            for c_start, c_end in chunks
        }
        for future in concurrent.futures.as_completed(futures):
            c_start, c_end = futures[future]
            try:
                chunk_scenes = future.result()
                all_scenes.extend(chunk_scenes)
            except Exception as e:
                print(f"    [find_matching_scenes] chunk [{c_start:.0f}-{c_end:.0f}s] skipped: {e}")

    return _merge_scenes_across_chunks(all_scenes)


# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------

# Safety cap so a huge folder tree can't make this hang forever.
MAX_FILES_SCANNED = 20000
# Folders to skip while walking — saves time, avoids irrelevant noise.
_SKIP_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "$RECYCLE.BIN",
    "System Volume Information", ".venv", "venv",
}

def _normalize(name: str) -> str:
    name = Path(name).stem  
    name = name.lower()
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
 
@tool
def locate_user_path(location_hint: str, target_name: str) -> str:
    """Use this tool when the user mentions a file or folder in a common location."""
    home = Path.home()
    hint = location_hint.lower()
    
    if "desktop" in hint: base = home / "Desktop"
    elif "download" in hint: base = home / "Downloads"
    elif "document" in hint: base = home / "Documents"
    elif "movie" in hint: base = home / "Movies"
    else: base = home
    
    if not base.exists():
        return json.dumps({"error": f"Location does not exist: {base}"})
 
    direct_path = base / target_name
    if direct_path.exists():
        return json.dumps({"absolute_path": str(direct_path)})
 
    target_norm = _normalize(target_name)
    scored_matches = []
    files_scanned = 0
 
    for p in base.rglob("*"):
        if files_scanned >= MAX_FILES_SCANNED:
            break
        if any(part in _SKIP_DIR_NAMES for part in p.parts):
            continue
        files_scanned += 1
 
        candidate_norm = _normalize(p.name)
        if target_norm == candidate_norm:
            return json.dumps({"absolute_path": str(p)})
        if target_norm and target_norm in candidate_norm:
            scored_matches.append(p)
 
    if scored_matches:
        scored_matches.sort(key=lambda p: len(p.name))
        best = scored_matches[0]
        if len(scored_matches) == 1:
            return json.dumps({"absolute_path": str(best)})
        return json.dumps({
            "absolute_path": str(best),
            "other_candidates": [str(p) for p in scored_matches[1:6]],
            "note": "Multiple possible matches found — confirm with the user.",
        })
 
    return json.dumps({"error": f"Could not find anything matching '{target_name}' in {base}."})

@tool
def list_video_clips(folder_path: str) -> str:
    """List video files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        return json.dumps({"error": f"Folder not found: {folder_path}"})
    if not folder.is_dir():
        return json.dumps({"error": f"Not a folder: {folder_path}"})

    video_files = [
        str(p) for p in folder.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
    ]
    if not video_files:
        return json.dumps({"error": f"No video files found in {folder_path}"})

    return json.dumps({"folder": folder_path, "video_files": video_files})

@tool
def evaluate_video_frame(video_path: str) -> str:
    """Score a single video clip's visual quality."""
    path = Path(video_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {video_path}"})

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frame_paths = _extract_frames(path, Path(tmp), count=3)
            images_b64 = [_encode_image_b64(fp) for fp in frame_paths]

        remote_payload = {"operation": "score", "images": images_b64, "n": len(images_b64)}
        parsed = _invoke_remote_scoring(remote_payload)
        parsed["video_path"] = video_path
        return json.dumps(parsed)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "video_path": video_path})

@tool
def find_matching_scenes(video_path: str, search_query: str) -> str:
    """Finds specific video scenes by extracting frames from a local video and comparing multimodal vectors."""
    import boto3
    from botocore.config import Config
    import json
    import numpy as np
    import cv2
    import base64
    import concurrent.futures
    import time
    from sklearn.metrics.pairwise import cosine_similarity
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return json.dumps({"error": f"Failed to open video file at {video_path}"})
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frames_b64 = []
    frame_timestamps = []
    
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % int(fps) == 0:
            resized = cv2.resize(frame, (1280, 720))
            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            video_frames_b64.append(b64_img)
            frame_timestamps.append(current_frame / fps)
            
        current_frame += 1
        
    cap.release()
    
    if not video_frames_b64:
        return json.dumps({"error": "Failed to extract any frames using OpenCV."})
        
    # Configure AWS for heavy rate-limiting with adaptive backoff
    retry_config = Config(
        retries={
            'max_attempts': 15,
            'mode': 'adaptive'
        }
    )
    client = boto3.client("bedrock-runtime", region_name="us-east-1", config=retry_config)
    model_id = "amazon.nova-2-multimodal-embeddings-v1:0" 
    
    try:
        text_payload = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingDimension": 1024,
                "embeddingPurpose": "GENERIC_INDEX",
                "text": {
                    "value": search_query,
                    "truncationMode": "END"
                }
            }
        }
        
        text_response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(text_payload),
            accept="application/json",
            contentType="application/json"
        )
        text_vector = np.array(json.loads(text_response['body'].read())['embeddings'][0]['embedding']).reshape(1, -1)
        
        total_frames = len(video_frames_b64)
        print(f"🎬 Extracting and embedding {total_frames} frames from {video_path}...")
        
        frame_vectors_dict = {}
        
        def embed_single_frame(index, b64_data):
            time.sleep(0.5) # Force a strict delay to respect AWS TPS quotas
            clean_b64 = b64_data.split(",")[-1] if "," in b64_data else b64_data
            img_payload = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingDimension": 1024,
                    "embeddingPurpose": "GENERIC_INDEX",
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": clean_b64}
                    }
                }
            }
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(img_payload),
                accept="application/json",
                contentType="application/json"
            )
            return index, json.loads(response['body'].read())['embeddings'][0]['embedding']

        # Drop workers to 2 to prevent overwhelming the Bedrock endpoint
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_index = {
                executor.submit(embed_single_frame, i, b64_img): i 
                for i, b64_img in enumerate(video_frames_b64)
            }
            
            processed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    _, vec = future.result()
                    frame_vectors_dict[idx] = vec
                except Exception as e:
                    print(f"⚙️ [Embedding Warning]: Frame {idx} failed: {str(e)}")
                    
                processed += 1
                if processed % 25 == 0 or processed == total_frames:
                    print(f"⏳ Embedded {processed}/{total_frames} frames...")

        if not frame_vectors_dict:
            return json.dumps({"error": "Failed to embed any frames via AWS Bedrock."})
            
        frame_vectors = []
        valid_timestamps = []
        for i in sorted(frame_vectors_dict.keys()):
            frame_vectors.append(frame_vectors_dict[i])
            valid_timestamps.append(frame_timestamps[i])
            
        frame_matrix = np.array(frame_vectors)
        
        similarities = cosine_similarity(text_vector, frame_matrix)[0]
        best_match_indices = np.where(similarities > 0.70)[0]
        
        if len(best_match_indices) == 0:
            best_match_indices = [np.argmax(similarities)]
            
        final_matches = [valid_timestamps[idx] for idx in best_match_indices]
        print(f"✅ Scene successfully detected at timestamps (seconds): {final_matches}")
        
        return json.dumps({"matched_timestamps_seconds": final_matches})
        
    except Exception as e:
        error_string = f"Embedding process failed: {str(e)}"
        print(f"❌ CRITICAL ERROR: {error_string}")
        return json.dumps({"error": error_string})
    
@tool
def process_audio_and_captions(video_or_draft_path: str, draft_name: str = "", spoken_query: str = "") -> str:
    """Transcribes audio from a single video file OR transcribes all clips across an entire CapCut draft timeline."""
    target_path = Path(video_or_draft_path)

    if target_path.is_dir() and draft_name:
        potential_target = target_path / draft_name
        if potential_target.exists():
            target_path = potential_target

    if not target_path.exists():
        return json.dumps({"error": f"Path not found: {target_path}"})

    model = whisper.load_model("base")

    if target_path.is_dir() or target_path.suffix == ".json":
        json_file = target_path / "draft_content.json" if target_path.is_dir() else target_path
        if not json_file.exists():
            return json.dumps({
                "error": f"draft_content.json not found in {target_path}."
            })

        try:
            draft_data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            return json.dumps({"error": f"Could not parse draft_content.json: {e}"})

        video_tracks = [t for t in draft_data.get("tracks", []) if t.get("type") == "video"]
        if not video_tracks or not video_tracks[0].get("segments"):
            return json.dumps({"error": "No video clips found on the draft timeline."})

        materials = {m["id"]: m["path"] for m in draft_data.get("materials", {}).get("videos", [])}
        all_segments = []
        timeline_offset = 0.0

        for seg in video_tracks[0]["segments"]:
            mat_id = seg.get("material_id")
            source_file = materials.get(mat_id)
            if not source_file or not Path(source_file).exists():
                continue

            src_range = seg.get("source_timerange", {})
            start_sec = src_range.get("start", 0) / 1_000_000.0
            dur_sec = src_range.get("duration", 0) / 1_000_000.0

            temp_wav = Path(tempfile.gettempdir()) / f"temp_seg_{uuid.uuid4().hex[:8]}.wav"
            cmd = ["ffmpeg", "-y", "-ss", str(start_sec)]
            if dur_sec > 0:
                cmd.extend(["-t", str(dur_sec)])
            cmd.extend(["-i", str(source_file), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)])

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            res = model.transcribe(str(temp_wav), fp16=False)
            for item in res.get("segments", []):
                all_segments.append({
                    "start": item["start"] + timeline_offset,
                    "end": item["end"] + timeline_offset,
                    "text": item["text"].strip()
                })

            timeline_offset += dur_sec
            if temp_wav.exists():
                temp_wav.unlink()

        out_srt = json_file.parent / f"{target_path.stem}_captions.srt"
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(all_segments, start=1):
                s_m, s_s = divmod(seg["start"], 60)
                s_h, s_m = divmod(s_m, 60)
                e_m, e_s = divmod(seg["end"], 60)
                e_h, e_m = divmod(e_m, 60)
                f.write(f"{i}\n{int(s_h):02d}:{int(s_m):02d}:{s_s:06.3f}".replace('.', ',') +
                        f" --> {int(e_h):02d}:{int(e_m):02d}:{e_s:06.3f}".replace('.', ',') +
                        f"\n{seg['text']}\n\n")

        full_text = " ".join(s["text"] for s in all_segments)
        return json.dumps({
            "status": "success",
            "srt_file": str(out_srt),
            "transcript_preview": full_text[:300],
            "message": f"Generated subtitles for all {len(video_tracks[0]['segments'])} timeline clips."
        })

    temp_wav = Path(tempfile.gettempdir()) / f"temp_raw_{uuid.uuid4().hex[:8]}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(target_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    result = model.transcribe(str(temp_wav), fp16=False)
    srt_path = target_path.parent / f"{target_path.stem}_captions.srt"
    
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            s_m, s_s = divmod(seg["start"], 60)
            s_h, s_m = divmod(s_m, 60)
            e_m, e_s = divmod(seg["end"], 60)
            e_h, e_m = divmod(e_m, 60)
            f.write(f"{i}\n{int(s_h):02d}:{int(s_m):02d}:{s_s:06.3f}".replace('.', ',') + 
                    f" --> {int(e_h):02d}:{int(e_m):02d}:{e_s:06.3f}".replace('.', ',') + 
                    f"\n{seg['text'].strip()}\n\n")

    if temp_wav.exists():
        temp_wav.unlink()

    return json.dumps({
        "status": "success",
        "srt_file": str(srt_path),
        "transcript_preview": result["text"][:300]
    })

@tool
def generate_b_roll_video(prompt: str, image_paths: list[str] = None) -> str:
    """Generate a 6-second video using Amazon Nova Reel based on a text prompt and optional starting/reference images."""
    import boto3
    import json
    import re
    import base64
    import io
    import time
    import os
    from pathlib import Path
    
    if image_paths is None:
        image_paths = []
        
    # Handle edge case where the agent passes a single string instead of a list
    if isinstance(image_paths, str):
        image_paths = [image_paths] if image_paths.strip() else []

    # 1. Strip hidden UI injections and extract any missed paths from the prompt
    clean_prompt = re.sub(r'\[System Note:.*?\]', '', prompt, flags=re.DOTALL).strip()
    extracted_paths = re.findall(r"located at '(.*?)'", prompt)
    
    for p in extracted_paths:
        if p not in image_paths:
            image_paths.append(p)
            
    if len(clean_prompt) > 512:
        clean_prompt = clean_prompt[:512]
        
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    s3_client = boto3.client("s3", region_name="us-east-1") 
    
    s3_bucket = "cdk-hnb659fds-assets-049584025361-us-east-1"
    s3_destination = f"s3://{s3_bucket}/nova_reel_outputs/"
    
    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": clean_prompt
        },
        "videoGenerationConfig": {
            "fps": 24,
            "durationSeconds": 6, 
            "dimension": "1280x720"
        }
    }
    
    # 2. Process Multiple Images
    images_array = []
    if image_paths:
        from PIL import Image
        for path in image_paths:
            if Path(path).exists():
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=95)
                        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        
                    images_array.append({
                        "format": "jpeg",
                        "source": {
                            "bytes": img_b64
                        }
                    })
                except Exception as e:
                    print(f"⚙️ [Image Warning]: Skipping {path} due to error: {str(e)}")
                    
        if images_array:
            model_input["textToVideoParams"]["images"] = images_array
            
    # 3. Trigger AWS Nova Reel & Poll for Completion
    try:
        response = client.start_async_invoke(
            modelId="amazon.nova-reel-v1:0",
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": s3_destination}}
        )
        invocation_arn = response["invocationArn"]
        
        print(f"🎬 Nova Reel rendering started using {len(images_array)} images. This usually takes 1-2 minutes...")
        
        elapsed_time = 0
        while True:
            time.sleep(10)
            elapsed_time += 10
            
            status_response = client.get_async_invoke(invocationArn=invocation_arn)
            job_status = status_response.get("status")
            
            if job_status == "Completed":
                print("✅ AWS render complete. Downloading MP4 to local machine...")
                
                objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix="nova_reel_outputs/")
                latest_mp4 = None
                
                if 'Contents' in objects:
                    mp4_files = [obj for obj in objects['Contents'] if obj['Key'].endswith('.mp4')]
                    if mp4_files:
                        latest_mp4 = max(mp4_files, key=lambda x: x['LastModified'])['Key']
                
                if latest_mp4:
                    local_path = os.path.join(os.getcwd(), "nova_generated_broll.mp4")
                    s3_client.download_file(s3_bucket, latest_mp4, local_path)
                    print(f"📥 Download successful: {local_path}")
                    
                    return json.dumps({
                        "status": "success",
                        "local_video_path": local_path,
                        "message": f"Tell the user the video is ready, then use {local_path} to create the CapCut draft."
                    })
                else:
                    return json.dumps({"error": "AWS job completed but no MP4 was found in the bucket."})
                    
            elif job_status == "Failed":
                error_msg = status_response.get("failureMessage", "Unknown error")
                print(f"❌ Video generation failed: {error_msg}")
                return f"AWS API Error: Job failed - {error_msg}"
                
            else:
                print(f"⏳ AWS Nova Reel is rendering... ({elapsed_time} seconds elapsed)")

    except Exception as e:
        return f"AWS API Error: {str(e)}"

@tool
def detect_and_trim_silence(video_path: str, noise_db: int = -30, min_duration: float = 0.5) -> str:
    """Analyzes a video file to find silence and returns a JSON array of the 'speaking' segments to keep."""
    target_path = Path(video_path)
    if not target_path.exists():
        return json.dumps({"error": f"File not found: {video_path}"})
        
    total_duration = _get_video_duration(target_path)

    cmd = [
        "ffmpeg", "-i", str(target_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    
    try:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stderr
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"FFmpeg processing failed: {e}"})

    silence_starts = []
    silence_ends = []
    
    for line in output.splitlines():
        if "silence_start" in line:
            match = re.search(r"silence_start:\s+([\d\.]+)", line)
            if match: silence_starts.append(float(match.group(1)))
        elif "silence_end" in line:
            match = re.search(r"silence_end:\s+([\d\.]+)", line)
            if match: silence_ends.append(float(match.group(1)))

    if len(silence_starts) > len(silence_ends):
        silence_ends.append(total_duration)

    keep_segments = []
    current_time = 0.0

    for start, end in zip(silence_starts, silence_ends):
        if start > current_time:
            keep_segments.append({
                "video_path": str(target_path),
                "start_sec": round(current_time, 3),
                "end_sec": round(start, 3)
            })
        current_time = end

    if current_time < total_duration:
        keep_segments.append({
            "video_path": str(target_path),
            "start_sec": round(current_time, 3),
            "end_sec": round(total_duration, 3)
        })

    return json.dumps({
        "status": "success",
        "total_silence_clips_removed": len(silence_starts),
        "keep_segments": keep_segments
    }, indent=2)

@tool
def find_capcut_drafts_folder() -> str:
    """Try to automatically locate the CapCut desktop app's drafts folder."""
    home = Path.home()
    candidates = [
        home / "AppData" / "Local" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
        home / "AppData" / "Local" / "CapCut" / "Apps",
        home / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
    ]

    found = [str(c) for c in candidates if c.exists() and c.is_dir()]
    windows_apps_root = home / "AppData" / "Local" / "CapCut" / "Apps"
    if windows_apps_root.exists():
        for match in windows_apps_root.glob("*/User Data/Projects/com.lveditor.draft"):
            if match.is_dir() and str(match) not in found:
                found.append(str(match))
    found = [f for f in found if f != str(windows_apps_root)] or found

    if not found:
        return json.dumps({"found": False, "message": "Could not auto-locate a CapCut drafts folder."})
    return json.dumps({"found": True, "candidates": found})

@tool
def create_capcut_draft(clip_paths: list[str], capcut_drafts_dir: str, music_path: str = "", is_vertical: bool = False, draft_name: str = "Agent_Draft") -> str:
    """Build a CapCut draft timeline from approved clip paths with optional background music."""
    if not clip_paths:
        return json.dumps({"error": "No clip paths provided"})
        
    canvas_width = 1080 if is_vertical else 1920
    canvas_height = 1920 if is_vertical else 1080

    draft_folder = cc.DraftFolder(capcut_drafts_dir)
    script = draft_folder.create_draft(draft_name, canvas_width, canvas_height, fps=30, allow_replace=True)
    video_track = script.add_track(cc.TrackType.video)
    
    current_start_us = 0
    included = []
    skipped = []

    for clip_path in clip_paths:
        p = Path(clip_path)
        if not p.exists():
            skipped.append({"clip": clip_path, "reason": "File not found"})
            continue

        p, fix_note, fix_error = _ensure_capcut_compatible(p)
        if fix_error:
            skipped.append({"clip": clip_path, "reason": fix_error})
            continue

        try:
            material = cc.VideoMaterial(str(p))
            duration = getattr(material, "duration", 3_000_000)
            segment = cc.VideoSegment(
                material, target_timerange=cc.Timerange(current_start_us, duration)
            )
            video_track.add_segment(segment)
            current_start_us += duration
            entry = {"clip": clip_path}
            if fix_note: entry["note"] = fix_note
            included.append(entry)
        except Exception as e:
            skipped.append({"clip": clip_path, "reason": str(e)})

    if music_path and included:
        mp = Path(music_path)
        if mp.exists():
            try:
                audio_track = script.add_track(cc.TrackType.audio)
                audio_material = cc.AudioMaterial(str(mp))
                audio_duration = min(getattr(audio_material, "duration", current_start_us), current_start_us)
                audio_segment = cc.AudioSegment(
                    audio_material,
                    target_timerange=cc.Timerange(0, audio_duration),
                    source_timerange=cc.Timerange(0, audio_duration),
                )
                audio_track.add_segment(audio_segment)
                included.append({"music": music_path})
            except Exception as e:
                skipped.append({"music": music_path, "reason": str(e)})
        else:
            skipped.append({"music": music_path, "reason": "Audio file not found"})

    if not included:
        return json.dumps({"error": "No clips could be added to draft", "skipped": skipped})

    script.save()
    draft_path = str(Path(capcut_drafts_dir) / draft_name)
    return json.dumps({
        "draft_name": draft_name,
        "draft_path": draft_path,
        "included": included,
        "skipped": skipped,
        "status": "saved"
    })

@tool
def extract_viral_hooks(video_path: str, num_hooks: int = 3, target_duration: int = 15, start_time_sec: float = 0.0, end_time_sec: float = None) -> str:
    """Analyzes a video file to find the most engaging segments for short-form social media."""
    from pathlib import Path
    import json
    import boto3
    import whisper
    import warnings
    
    warnings.filterwarnings("ignore")
    
    video_file = Path(video_path)
    if not video_file.exists():
        return json.dumps({"error": f"Video file not found at {video_path}"})
        
    # 1. INJECT TIMESTAMPS INTO THE TRANSCRIPT
    print(f"🎙️ Transcribing audio from {video_file.name} using Whisper...")
    try:
        whisper_model = whisper.load_model("base")
        transcription_result = whisper_model.transcribe(str(video_file))
        
        # Loop through segments to create a timestamped script for the LLM
        formatted_transcript = ""
        for seg in transcription_result["segments"]:
            start = seg['start']
            end = seg['end']
            text = seg['text'].strip()
            formatted_transcript += f"[{start:.2f}s - {end:.2f}s] {text}\n"
            
        print("✅ Timestamped transcription complete. Analyzing for viral hooks...")
    except Exception as e:
        return json.dumps({"error": f"Whisper transcription failed: {str(e)}"})

    if not formatted_transcript.strip():
        return json.dumps({"error": "No dialogue found in the video to analyze."})
    
    # 2. UPGRADED SYSTEM PROMPT (Forced EXACT count, simplified time extraction)
    system_prompt = (
        "You are an elite short-form video editor specializing in viral content curation. "
        "Your task is to analyze this timestamped video transcript and extract the absolute most engaging clips. "
        "Aggressively filter out slow build-ups, boring exposition, and dead air. "
        "Hunt for universal retention triggers: "
        "1) High emotional peaks (shock, laughter, intense reactions). "
        "2) Strong curiosity gaps, bold claims, or contrarian statements. "
        "3) Mind-blowing facts or rapid pacing shifts. "
        "You must return EXACTLY a strict JSON array of objects, with no markdown formatting (no ```json). "
        "Each object must have exactly five keys: "
        "'hook_title', "
        "'start_sec' (float, the exact start time in seconds based on the transcript brackets), "
        "'end_sec' (float, the start time plus the target duration), "
        "'viral_score' (1-100, rank them, but you MUST output exactly the requested number of hooks regardless of score), "
        "and 'reasoning' (explain the specific retention trigger used)."
    )
    
    user_prompt = (
        f"You MUST extract EXACTLY {num_hooks} viral hooks from this transcript. "
        f"Target duration for each hook: exactly {target_duration} seconds.\n\n"
        f"Timestamped Transcript:\n{formatted_transcript}"
    )
    
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    try:
        response = client.converse(
            modelId="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.3} 
        )
        
        llm_response = response['output']['message']['content'][0]['text']
        llm_response = llm_response.replace('```json', '').replace('```', '').strip()
        hooks_data = json.loads(llm_response)
        
        # We removed the messy HH:MM:SS math. The LLM now just outputs raw seconds (e.g. 45.5)
        for hook in hooks_data:
            if 'start_sec' in hook:
                hook['end_sec'] = hook['start_sec'] + target_duration
                
        return json.dumps({
            "status": "success",
            "source_video": str(video_file),
            "target_duration_sec": target_duration,
            "hooks": hooks_data
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to extract hooks: {str(e)}"})
    

@tool
def add_scenes_to_existing_draft(scenes: list[dict], capcut_drafts_dir: str, draft_name: str) -> str:
    """Appends new video scenes to the END of an already existing CapCut draft timeline."""
    base_dir = Path(capcut_drafts_dir)
    draft_folder = base_dir / draft_name if base_dir.name != draft_name else base_dir
    json_file = draft_folder / "draft_content.json"
    
    if not json_file.exists():
        return json.dumps({"error": f"Draft not found at {draft_folder}"})

    if not scenes:
        return json.dumps({"error": "No scenes provided"})

    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        
        main_track = next((t for t in data.get("tracks", []) if t.get("type") == "video"), None)
        if not main_track:
            return json.dumps({"error": "No video track found in the existing draft to append to."})

        current_start_us = 0
        for seg in main_track.get("segments", []):
            tr = seg.get("target_timerange", {})
            seg_end = tr.get("start", 0) + tr.get("duration", 0)
            if seg_end > current_start_us:
                current_start_us = seg_end

        included = []
        skipped = []

        if "materials" not in data: data["materials"] = {}
        if "videos" not in data["materials"]: data["materials"]["videos"] = []

        for scene in scenes:
            video_path = scene.get("video_path")
            start_sec = scene.get("start_sec")
            end_sec = scene.get("end_sec")
            
            if not video_path or start_sec is None or end_sec is None:
                skipped.append({"scene": scene, "reason": "Missing required time/path data"})
                continue

            p = Path(video_path)
            if not p.exists():
                skipped.append({"scene": scene, "reason": "File not found"})
                continue

            source_start_us = int(start_sec * 1_000_000)
            duration_us = int((end_sec - start_sec) * 1_000_000)
            if duration_us <= 0: continue

            material_id = str(uuid.uuid4()).replace("-", "")
            segment_id = str(uuid.uuid4()).replace("-", "")

            video_material = {
                "id": material_id,
                "path": str(p.resolve()),
                "duration": int(_get_video_duration(p) * 1_000_000),
                "type": "video",
                "file_Path": str(p.resolve()),
                "name": p.name
            }
            data["materials"]["videos"].append(video_material)

            video_segment = {
                "id": segment_id,
                "material_id": material_id,
                "target_timerange": {"start": current_start_us, "duration": duration_us},
                "source_timerange": {"start": source_start_us, "duration": duration_us},
                "speed": 1.0,
                "volume": 1.0
            }
            main_track["segments"].append(video_segment)
            
            current_start_us += duration_us
            included.append(scene)

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json.dumps({"status": "success", "draft_name": draft_name, "scenes_added": included, "skipped": skipped})
    except Exception as e:
        return json.dumps({"error": f"Failed to add scenes to existing draft: {str(e)}"})
    
    
@tool
def create_capcut_draft_from_scenes(scenes: list[dict], capcut_drafts_dir: str, music_path: str = "", is_vertical: bool = False, draft_name: str = "Agent_Draft") -> str:
    """Build a CapCut draft timeline from trimmed matching SUB-SCENES with optional music."""
    if not scenes: return json.dumps({"error": "No scenes provided"})
        
    canvas_width = 1080 if is_vertical else 1920
    canvas_height = 1920 if is_vertical else 1080

    try:
        draft_folder = cc.DraftFolder(capcut_drafts_dir)
        script = draft_folder.create_draft(draft_name, canvas_width, canvas_height, fps=30, allow_replace=True)
        video_track = script.add_track(cc.TrackType.video)
    except Exception as e:
        return json.dumps({"error": str(e)})

    current_start_us = 0
    included = []
    skipped = []

    for scene in scenes:
        video_path = scene.get("video_path")
        start_sec = scene.get("start_sec")
        end_sec = scene.get("end_sec")
        if not video_path or start_sec is None or end_sec is None or end_sec <= start_sec:
            skipped.append({"scene": scene, "reason": "Invalid parameters"})
            continue

        p = Path(video_path)
        if not p.exists():
            skipped.append({"scene": scene, "reason": f"File not found: {video_path}"})
            continue

        p, fix_note, fix_error = _ensure_capcut_compatible(p)
        if fix_error:
            skipped.append({"scene": scene, "reason": fix_error})
            continue

        source_start_us = int(start_sec * 1_000_000)
        duration_us = int((end_sec - start_sec) * 1_000_000)

        try:
            material = cc.VideoMaterial(str(p))
            material_duration_us = getattr(material, "duration", None)
            if material_duration_us:
                safety_margin_us = 1000
                max_end_us = max(0, material_duration_us - safety_margin_us)
                if source_start_us >= max_end_us:
                    skipped.append({"scene": scene, "reason": "Scene starts after clip ends"})
                    continue
                clamped_end_us = min(source_start_us + duration_us, max_end_us)
                duration_us = clamped_end_us - source_start_us

            if duration_us <= 0: continue

            segment = cc.VideoSegment(
                material,
                target_timerange=cc.Timerange(current_start_us, duration_us),
                source_timerange=cc.Timerange(source_start_us, duration_us),
            )
            video_track.add_segment(segment)
            current_start_us += duration_us
            if fix_note: scene = dict(scene, note=fix_note)
            included.append(scene)
        except Exception as e:
            skipped.append({"scene": scene, "reason": str(e)})

    if music_path and included:
        mp = Path(music_path)
        if mp.exists():
            try:
                audio_track = script.add_track(cc.TrackType.audio)
                audio_material = cc.AudioMaterial(str(mp))
                audio_duration = min(getattr(audio_material, "duration", current_start_us), current_start_us)
                audio_segment = cc.AudioSegment(
                    audio_material,
                    target_timerange=cc.Timerange(0, audio_duration),
                    source_timerange=cc.Timerange(0, audio_duration),
                )
                audio_track.add_segment(audio_segment)
                included.append({"music": music_path})
            except Exception as e:
                skipped.append({"music": music_path, "reason": str(e)})
        else:
            skipped.append({"music": music_path, "reason": "Audio file not found"})

    if not included:
        return json.dumps({"error": "No scenes added to draft", "skipped": skipped})

    script.save()
    draft_path = str(Path(capcut_drafts_dir) / draft_name)
    return json.dumps({
        "draft_name": draft_name,
        "draft_path": draft_path,
        "scenes_included": included,
        "skipped": skipped,
        "status": "saved"
    })

@tool
def add_music_to_existing_draft(capcut_drafts_dir: str, draft_name: str, music_path: str) -> str:
    """Adds background music to an already existing CapCut draft timeline."""
    base_dir = Path(capcut_drafts_dir)
    draft_folder = base_dir / draft_name if base_dir.name != draft_name else base_dir
    json_file = draft_folder / "draft_content.json"
    
    if not json_file.exists():
        return json.dumps({"error": f"Draft not found at {draft_folder}"})

    mp = Path(music_path)
    if not mp.exists():
        return json.dumps({"error": f"Music file not found: {music_path}"})

    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        
        total_duration_us = 0
        for track in data.get("tracks", []):
            if track.get("type") == "video":
                for seg in track.get("segments", []):
                    tr = seg.get("target_timerange", {})
                    seg_end = tr.get("start", 0) + tr.get("duration", 0)
                    if seg_end > total_duration_us:
                        total_duration_us = seg_end

        if total_duration_us == 0:
            total_duration_us = 10_000_000

        audio_dur_sec = _get_video_duration(mp)
        audio_dur_us = int(audio_dur_sec * 1_000_000)
        target_audio_dur = min(audio_dur_us, total_duration_us)

        material_id = str(uuid.uuid4()).replace("-", "")
        track_id = str(uuid.uuid4()).replace("-", "")
        segment_id = str(uuid.uuid4()).replace("-", "")

        audio_material = {
            "id": material_id,
            "path": str(mp.resolve()),
            "duration": audio_dur_us,
            "type": "audio",
            "file_Path": str(mp.resolve()),
            "name": mp.name
        }

        if "materials" not in data: data["materials"] = {}
        if "audios" not in data["materials"]: data["materials"]["audios"] = []
        data["materials"]["audios"].append(audio_material)

        audio_segment = {
            "id": segment_id,
            "material_id": material_id,
            "target_timerange": {"start": 0, "duration": target_audio_dur},
            "source_timerange": {"start": 0, "duration": target_audio_dur},
            "speed": 1.0,
            "volume": 1.0
        }

        audio_track = {
            "id": track_id,
            "type": "audio",
            "segments": [audio_segment]
        }

        if "tracks" not in data: data["tracks"] = []
        data["tracks"] = [t for t in data["tracks"] if t.get("type") != "audio"]
        data["tracks"].append(audio_track)

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json.dumps({
            "status": "success",
            "draft_name": draft_name,
            "draft_path": str(draft_folder),
            "message": f"Successfully added background music {mp.name} to existing draft '{draft_name}'."
        })
    except Exception as e:
        return json.dumps({"error": f"Failed to add music to existing draft: {str(e)}"})


# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are an interactive, helpful AI post-production coordinator. 

If the user mentions a folder or file using casual language (like 'it is on my desktop' or 'in downloads'), DO NOT ask them for the absolute path. Instead, use the locate_user_path tool to find it yourself.

When you have the absolute path to a clips folder, ALWAYS call list_video_clips on it first to discover what's actually in there before doing anything else.

When creating CapCut drafts, generate a concise, descriptive 'draft_name' based on the user's request (e.g., 'Arcane_Vertical_Hooks' or 'Silence_Trimmed_Edit').

You have multiple main workflows:

WORKFLOW 1 — General quality curation:
1. Evaluate every clip using evaluate_video_frame.
2. Decide which clips to include (favoring scores >= 6).
3. Call create_capcut_draft with the approved clips.
4. Report back: which clips were included, which were rejected, and draft confirmation.

WORKFLOW 2 — Content-directed visual scene search:
1. Call find_matching_scenes on EVERY clip in the folder with the user's content_query.
2. Collect all matching sub-scenes across all clips into a single ordered list.
3. Call create_capcut_draft_from_scenes with that combined list.
4. Report back which clips contributed scenes and confirm the draft was saved.

WORKFLOW 3 — Speech-to-text, dialogue search & auto-captions:
1. When asked to generate subtitles/captions for a draft or timeline, DO NOT ask the user for video paths.
2. Use find_capcut_drafts_folder to locate the CapCut root drafts directory.
3. Call process_audio_and_captions passing the CapCut root drafts directory as 'video_or_draft_path', AND pass the project's exact name as 'draft_name'.
4. Inform the user that an .srt subtitle file has been generated inside the project directory and can be dragged onto the CapCut timeline.

WORKFLOW 4 — Viral hook extraction & vertical social edits (TikTok/Shorts):
1. When asked to find hooks, call extract_viral_hooks on the video file. 
2. If the user specifies how long the hooks should be, pass 'target_duration'. 
3. If the user specifies a time constraint (e.g., "after 1 hour 20 mins"), calculate the total seconds (e.g., 4800) and pass 'start_time_sec' and/or 'end_time_sec'.
4. Format the returned hooks as a list of dictionaries with 'video_path', 'start_sec', and 'end_sec' keys.
5. Pass that list into create_capcut_draft_from_scenes and explicitly set is_vertical=True to enforce the 9:16 vertical crop.

WORKFLOW 5 — Dead Air & Silence Trimming (Jump-Cuts):
1. When asked to remove silence or dead air, call detect_and_trim_silence on the video.
2. Take the returned 'keep_segments' (which include 'video_path', 'start_sec', and 'end_sec').
3. Pass them to create_capcut_draft_from_scenes to automatically assemble a jump-cut timeline.

WORKFLOW 6 — Adding music to an existing draft:
1. When asked to add background music to an already created draft, use find_capcut_drafts_folder to get the root drafts directory.
2. Use locate_user_path if the song file is in Downloads or Desktop.
3. Call add_music_to_existing_draft with the exact draft_name, capcut_drafts_dir, and music_path.

WORKFLOW 7 — Adding new video scenes to an existing draft:
1. When asked to add a specific visual scene or video clip to a draft that already exists, use find_capcut_drafts_folder first.
2. Call find_matching_scenes on the video file to locate the requested footage.
3. Pass the resulting scenes into add_scenes_to_existing_draft along with the exact draft_name.

BACKGROUND MUSIC (during draft creation):
If the user asks to add background music during initial draft creation, ask them for the file name/location, use locate_user_path to find it, and pass that path to the music_path parameter in create_capcut_draft or create_capcut_draft_from_scenes.

For the CapCut drafts folder: ALWAYS call find_capcut_drafts_folder first. If it finds one candidate, use it directly.

Clips using HEVC/H.265 or 10-bit color are automatically re-encoded to 8-bit H.264. You can mention this in passing if noted in the tool result.
"""

def _clean_agent_output(text: str) -> str:
    cleaned = re.sub(r"<thinking>.*?</thinking>\s*", "", str(text), flags=re.DOTALL)
    cleaned = re.sub(r"</?response>\s*", "", cleaned)
    return cleaned.strip()

def run_chat_loop(agent):
    print("🎬 Post-Production Assistant ready. Type your request, or 'exit' to quit.")
    print('Example: "Make a draft from the clips folder on my desktop and add lo-fi.mp3 as background music"\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        try:
            response = agent(user_input)
            print(f"\nAgent: {_clean_agent_output(response)}\n")
        except Exception as e:
            print(f"\n[Error during agent turn: {type(e).__name__}: {e}]\n")



agent = Agent(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            find_capcut_drafts_folder,
            list_video_clips,
            locate_user_path,
            evaluate_video_frame,
            find_matching_scenes,
            process_audio_and_captions,
            create_capcut_draft,
            create_capcut_draft_from_scenes,
            extract_viral_hooks,
            detect_and_trim_silence,
            add_music_to_existing_draft,
            add_scenes_to_existing_draft,
            generate_b_roll_video
        ],
    )


def main():
    if len(sys.argv) >= 3:
        clips_folder = Path(sys.argv[1])
        capcut_drafts_dir = sys.argv[2]
        content_query = sys.argv[3] if len(sys.argv) == 4 else None

        if not clips_folder.exists():
            print(f"Folder not found: {clips_folder}")
            sys.exit(1)

        video_files = [
            str(p) for p in clips_folder.iterdir()
            if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
        ]
        if not video_files:
            print(f"No video files found in {clips_folder}")
            sys.exit(1)

        print(f"Found {len(video_files)} clip(s): {[Path(v).name for v in video_files]}")

        if content_query:
            prompt = (
                f"Here are the raw video clips: {video_files}\n"
                f"CapCut drafts folder: {capcut_drafts_dir}\n"
                f"Content query: \"{content_query}\"\n"
                "Use WORKFLOW 2 to search for matching scenes and build the draft."
            )
        else:
            prompt = (
                f"Here are the raw video clips: {video_files}\n"
                f"CapCut drafts folder: {capcut_drafts_dir}\n"
                "Use WORKFLOW 1 to evaluate quality and build the draft."
            )

        result = agent(prompt)
        print("\n--- Agent final response ---")
        print(_clean_agent_output(result))
    else:
        run_chat_loop(agent)

if __name__ == "__main__":
    main()