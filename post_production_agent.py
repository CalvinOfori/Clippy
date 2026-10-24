"""
post_production_agent.py  (NVIDIA NIM edition)

Strands agent for the Multi-Modal Post-Production Assistant. Runs LOCALLY
(needs local ffmpeg, local files and the CapCut desktop drafts folder).

Models:
  - Orchestrator (tool calling) + hook extraction:
        AWS Bedrock nvidia.nemotron-nano-12b-v2   (ORCHESTRATOR=bedrock, default)
        or an NVIDIA NIM text model                (ORCHESTRATOR=nim)
  - Vision (scoring / scene search): NVIDIA NIM Llama 3.2 Vision

Environment variables:
  NVIDIA_API_KEY      required (vision calls)
  ORCHESTRATOR        bedrock | nim   (default bedrock)
  BEDROCK_MODEL_ID    default nvidia.nemotron-nano-12b-v2
  AWS_REGION          default us-east-1
  TEXT_MODEL_ID       only if ORCHESTRATOR=nim (no default; set it to a NIM model id)
  VISION_MODEL_ID     default meta/llama-3.2-90b-vision-instruct
  CAPCUT_DRAFTS_DIR   optional override of the CapCut drafts root folder
"""

import base64
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import pycapcut as cc
import whisper
from openai import OpenAI
from strands import Agent, tool

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY ="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"
VISION_MODEL_ID = os.environ.get("VISION_MODEL_ID", "meta/llama-3.2-90b-vision-instruct")
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "")  # only used when ORCHESTRATOR=nim
ORCHESTRATOR = os.environ.get("ORCHESTRATOR", "bedrock").lower()
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "nvidia.nemotron-nano-12b-v2")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Hard timeouts: a stalled NIM call must fail fast instead of hanging the whole agent.
nvidia_client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY, timeout=40.0, max_retries=0)

MAX_SAMPLE_FRAMES = 40
MIN_SAMPLE_INTERVAL_SEC = 0.5
MAX_VISION_WORKERS = 3
_REENCODE_CACHE_DIR = Path(tempfile.gettempdir()) / "post_production_agent_reencoded"


def _build_model():
    """Orchestrator model for the Strands agent (default: Nemotron Nano on Bedrock, pinned to AWS_REGION)."""
    if ORCHESTRATOR == "bedrock":
        from strands.models import BedrockModel
        return BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0.2)
    if not TEXT_MODEL_ID:
        raise RuntimeError("ORCHESTRATOR=nim needs TEXT_MODEL_ID set to a NIM model id.")
    from strands.models.openai import OpenAIModel  # pip install "strands-agents[openai]"
    return OpenAIModel(
        client_args={"api_key": NVIDIA_API_KEY, "base_url": NIM_BASE_URL},
        model_id=TEXT_MODEL_ID,
        params={"temperature": 0.2, "max_tokens": 2048},
    )


# ---------------------------------------------------------------------
# NIM HELPERS
# ---------------------------------------------------------------------
def _nim_chat(model: str, messages: list, temperature: float = 0.0, max_tokens: int = 512, retries: int = 2) -> str:
    last = None
    for attempt in range(retries):
        try:
            r = nvidia_client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return r.choices[0].message.content or ""
        except Exception as e:  # rate limits, timeouts
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


_bedrock_client = None


def _text_llm(system: str, user: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """Text reasoning call: Bedrock (default, Nemotron Nano) or NVIDIA NIM, matching ORCHESTRATOR."""
    global _bedrock_client
    if ORCHESTRATOR == "bedrock":
        if _bedrock_client is None:
            import boto3
            _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        resp = _bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user}]}],
            system=[{"text": system}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        # Reasoning models may return a reasoning block first; keep only the text blocks.
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    return _nim_chat(TEXT_MODEL_ID,
                     [{"role": "system", "content": system}, {"role": "user", "content": user}],
                     temperature=temperature, max_tokens=max_tokens)


def _parse_json(text: str):
    """Pull the first JSON object/array out of an LLM reply."""
    cleaned = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text or "", flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON in model reply: {cleaned[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[min(starts):])
    return obj


def _vision_ask(prompt: str, image_b64: str, max_tokens: int = 300) -> str:
    clean = image_b64.split(",")[-1] if "," in image_b64 else image_b64
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean}"}},
    ]
    # NIM vision models take ONE image per request, so we always call per frame.
    return _nim_chat(VISION_MODEL_ID, [{"role": "user", "content": content}], 0.0, max_tokens)


# ---------------------------------------------------------------------
# LOCAL FFMPEG & MEDIA HELPERS
# ---------------------------------------------------------------------
def _get_video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        try:
            result = subprocess.run(["ffmpeg", "-i", str(video_path)], capture_output=True, text=True)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass
        return 10.0


def _extract_frame_at(video_path: Path, timestamp: float, out_path: Path, width: int = 768) -> None:
    def _run(ts: float):
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out_path)],
            capture_output=True, check=True,
        )
    try:
        _run(timestamp)
    except subprocess.CalledProcessError:
        _run(max(0.0, timestamp - 0.5))


def _extract_frames(video_path: Path, out_dir: Path, count: int = 3) -> list[Path]:
    duration = _get_video_duration(video_path)
    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    paths = []
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"frame_{i}.jpg"
        _extract_frame_at(video_path, ts, out_path)
        paths.append(out_path)
    return paths


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
    return [min(duration * i / (n - 1), safe_max) for i in range(n)]


def _merge_matches_into_scenes(matches: list[int], timestamps: list[float], duration: float) -> list[dict]:
    matches = sorted({m for m in matches if 0 <= m < len(timestamps)})
    if not matches:
        return []
    interval = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else duration
    half_pad = interval / 2

    runs, run_start, prev = [], matches[0], matches[0]
    for idx in matches[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((run_start, prev))
        run_start = prev = idx
    runs.append((run_start, prev))

    scenes = []
    for s_idx, e_idx in runs:
        t_start = max(0.0, timestamps[s_idx] - half_pad)
        t_end = min(duration, timestamps[e_idx] + half_pad)
        scenes.append({"start_sec": round(t_start, 2), "end_sec": round(t_end, 2)})
    return scenes


# --- CapCut codec compatibility (HEVC / 10-bit clips are re-encoded) ---
def _get_codec_info(video_path: Path) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt", "-of", "default=noprint_wrappers=1", str(video_path)],
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
    codec_name, pix_fmt = info.get("codec_name", ""), info.get("pix_fmt", "")
    reasons = []
    if codec_name in ("hevc", "h265"):
        reasons.append(f"codec is {codec_name.upper()} (HEVC/H.265)")
    if "10le" in pix_fmt or "10be" in pix_fmt or "p010" in pix_fmt:
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
    """Returns (path_to_use, note, error)."""
    info = _get_codec_info(video_path)
    if not info:
        return video_path, None, None
    reasons = _incompatibility_reasons(info)
    if not reasons:
        return video_path, None, None

    reason_str = ", ".join(reasons)
    cache_path = _reencoded_cache_path(video_path)
    if cache_path.exists():
        return cache_path, f"{video_path.name}: auto-re-encoded ({reason_str}) - reused cached 8-bit copy", None

    print(f"    [compat] {video_path.name}: {reason_str} - auto re-encoding...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k", str(cache_path)],
            capture_output=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr)
        cache_path.unlink(missing_ok=True)
        return video_path, None, f"{video_path.name} needs re-encoding ({reason_str}) but ffmpeg failed: {stderr[:300]}"
    except FileNotFoundError:
        return video_path, None, f"{video_path.name} needs re-encoding ({reason_str}) but ffmpeg is not on PATH"
    return cache_path, f"{video_path.name}: auto-re-encoded to 8-bit H.264 ({reason_str})", None


# ---------------------------------------------------------------------
# CAPCUT DRAFT HELPERS  (the code, not the LLM, owns paths + names)
# ---------------------------------------------------------------------
_GENERIC_NAMES = {"", "agent_draft", "draft", "untitled", "none", "null", "new_draft", "my_draft"}


def _drafts_root() -> Path:
    """The CapCut drafts ROOT folder. Draft folders are created INSIDE this."""
    override = os.environ.get("CAPCUT_DRAFTS_DIR", "").strip()
    if override:
        return Path(override)
    home = Path.home()
    candidates = [
        home / "AppData" / "Local" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
        home / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft",
    ]
    apps = home / "AppData" / "Local" / "CapCut" / "Apps"
    if apps.exists():
        candidates += list(apps.glob("*/User Data/Projects/com.lveditor.draft"))
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "CapCut drafts folder not found. Open CapCut once, or set CAPCUT_DRAFTS_DIR to the "
        "'com.lveditor.draft' folder."
    )


def _resolve_draft_name(root: Path, draft_name: str = "", hint: str = "") -> str:
    """Clean, Windows-safe, UNIQUE name. Never collides with (or overwrites) an existing draft."""
    name = (draft_name or "").strip()
    # The model sometimes passes a full path or a doubled name - keep only the last part.
    name = re.split(r"[\\/]", name)[-1].strip()
    if name.lower() in _GENERIC_NAMES:
        base = re.sub(r"[^\w\- ]+", "", hint or "").strip().replace(" ", "_")[:30] or "Clippy_Draft"
        name = f"{base}_{datetime.now():%m%d_%H%M}"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")[:60] or "Clippy_Draft"

    candidate, n = name, 2
    while (root / candidate).exists():
        candidate = f"{name}_{n}"
        n += 1
    return candidate


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_existing_draft(draft_name: str) -> Path | None:
    """Exact folder match first, then a forgiving match. Only folders that contain a draft."""
    root = _drafts_root()
    direct = root / draft_name
    if (direct / "draft_content.json").exists():
        return direct
    target = _norm(draft_name)
    for d in root.iterdir():
        if d.is_dir() and (d / "draft_content.json").exists() and _norm(d.name) == target:
            return d
    return None


def _attach_music(script, music_path: str, total_us: int):
    """Returns (music_entry, skipped_entry) - exactly one is not None."""
    mp = Path(music_path)
    if not mp.exists():
        return None, {"music": music_path, "reason": "Audio file not found"}
    try:
        audio_track = script.add_track(cc.TrackType.audio)
        audio_material = cc.AudioMaterial(str(mp))
        audio_duration = min(getattr(audio_material, "duration", total_us), total_us)
        audio_track.add_segment(cc.AudioSegment(
            audio_material,
            target_timerange=cc.Timerange(0, audio_duration),
            source_timerange=cc.Timerange(0, audio_duration),
        ))
        return {"music": music_path}, None
    except Exception as e:
        return None, {"music": music_path, "reason": str(e)}


def _finish_draft(script, root: Path, name: str, extra: dict) -> str:
    script.save()
    draft_dir = root / name
    verified = (draft_dir / "draft_content.json").exists()
    out = {
        "draft_name": name,
        "draft_path": str(draft_dir),
        "verified_on_disk": verified,
        "status": "saved" if verified else "saved_but_draft_content_json_missing",
        "note": "Restart/refresh CapCut's project list if the draft is not visible yet.",
        **extra,
    }
    return json.dumps(out)


# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------
MAX_FILES_SCANNED = 20000
_SKIP_DIR_NAMES = {"node_modules", ".git", "__pycache__", "$RECYCLE.BIN",
                   "System Volume Information", ".venv", "venv"}


def _normalize(name: str) -> str:
    name = Path(name).stem.lower()
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


@tool
def locate_user_path(location_hint: str, target_name: str) -> str:
    """Use this tool when the user mentions a file or folder in a common location (desktop, downloads, documents)."""
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
    scored_matches, files_scanned = [], 0
    for p in base.rglob("*"):
        if files_scanned >= MAX_FILES_SCANNED:
            break
        if any(part in _SKIP_DIR_NAMES for part in p.parts):
            continue
        files_scanned += 1
        cand = _normalize(p.name)
        if target_norm == cand:
            return json.dumps({"absolute_path": str(p)})
        if target_norm and target_norm in cand:
            scored_matches.append(p)

    if scored_matches:
        scored_matches.sort(key=lambda p: len(p.name))
        best = scored_matches[0]
        if len(scored_matches) == 1:
            return json.dumps({"absolute_path": str(best)})
        return json.dumps({
            "absolute_path": str(best),
            "other_candidates": [str(p) for p in scored_matches[1:6]],
            "note": "Multiple possible matches found - confirm with the user.",
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
    video_files = [str(p) for p in folder.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    if not video_files:
        return json.dumps({"error": f"No video files found in {folder_path}"})
    return json.dumps({"folder": folder_path, "video_files": video_files})


@tool
def evaluate_video_frame(video_path: str) -> str:
    """Score a single video clip's visual quality (1-10) using NVIDIA NIM vision."""
    path = Path(video_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {video_path}"})

    prompt = ("Rate the visual quality of this video frame (sharpness, exposure, composition, noise). "
              "Reply with ONLY JSON: {\"score\": <number 1-10>, \"tags\": [<short strings>], \"notes\": \"<one sentence>\"}")

    def _score(b64: str):
        try:
            d = _parse_json(_vision_ask(prompt, b64))
            return float(d.get("score")), d.get("tags") or [], d.get("notes") or ""
        except Exception as e:
            return None, [], f"{type(e).__name__}: {e}"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = _extract_frames(path, Path(tmp), count=3)
            images = [_encode_image_b64(fp) for fp in frames]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VISION_WORKERS) as pool:
            results = list(pool.map(_score, images))
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "video_path": video_path})

    scores = [r[0] for r in results if r[0] is not None]
    if not scores:
        return json.dumps({"error": f"Vision scoring failed: {results[0][2]}", "video_path": video_path})
    tags = sorted({t for r in results for t in r[1] if isinstance(t, str)})
    notes = next((r[2] for r in results if r[2]), "")
    return json.dumps({
        "video_path": video_path,
        "score": round(sum(scores) / len(scores), 1),
        "tags": tags,
        "notes": notes,
        "frames_scored": len(scores),
    })


@tool
def find_matching_scenes(video_path: str, search_query: str) -> str:
    """Find scenes in ONE video matching a visual description. Returns ready-to-use 'scenes' (video_path, start_sec, end_sec)."""
    path = Path(video_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {video_path}"})

    duration = _get_video_duration(path)
    timestamps = _compute_sample_timestamps(duration)
    prompt = (f"Does this video frame show: \"{search_query}\"? "
              "Reply with ONLY JSON: {\"match\": true} or {\"match\": false}")

    def _check(i_ts):
        i, ts = i_ts
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fp = Path(tmp) / "f.jpg"
                _extract_frame_at(path, ts, fp)
                d = _parse_json(_vision_ask(prompt, _encode_image_b64(fp), max_tokens=60))
            return i, str(d.get("match")).lower() == "true", None
        except Exception as e:
            return i, False, f"{type(e).__name__}: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VISION_WORKERS) as pool:
        results = list(pool.map(_check, list(enumerate(timestamps))))

    failed = [r for r in results if r[2]]
    if len(failed) == len(results):
        return json.dumps({"error": f"Vision search failed on every frame: {failed[0][2]}"})

    matches = [i for i, ok, _ in results if ok]
    scenes = [dict(s, video_path=str(path)) for s in _merge_matches_into_scenes(matches, timestamps, duration)]
    return json.dumps({
        "video_path": str(path),
        "scenes": scenes,
        "matched_timestamps_seconds": [round(timestamps[i], 2) for i in matches],
        "frames_checked": len(results),
        "frames_failed": len(failed),
        "note": "" if scenes else "No frames matched this description.",
    })


@tool
def extract_viral_hooks(video_path: str, num_hooks: int = 3, target_duration: int = 15,
                        start_time_sec: float = 0.0, end_time_sec: float = None) -> str:
    """Transcribe a video and pick the most engaging segments for short-form social media. Returns ready-to-use 'scenes'."""
    import warnings
    warnings.filterwarnings("ignore")

    video_file = Path(video_path)
    if not video_file.exists():
        return json.dumps({"error": f"Video file not found at {video_path}"})

    print(f"Transcribing {video_file.name} with Whisper...")
    try:
        result = whisper.load_model("base").transcribe(str(video_file))
        lines = []
        for seg in result["segments"]:
            if seg["start"] < (start_time_sec or 0.0):
                continue
            if end_time_sec is not None and seg["end"] > end_time_sec:
                continue
            lines.append(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'].strip()}")
        transcript = "\n".join(lines)
    except Exception as e:
        return json.dumps({"error": f"Whisper transcription failed: {e}"})

    if not transcript.strip():
        return json.dumps({"error": "No dialogue found in the requested range."})

    system_prompt = (
        "You are an elite short-form video editor specializing in viral content curation. "
        "Analyze the timestamped transcript and pick the most engaging clips: emotional peaks, "
        "curiosity gaps, bold or contrarian claims, surprising facts. Skip slow build-ups and dead air. "
        "Return ONLY a JSON array (no markdown) of objects with exactly these keys: "
        "'hook_title', 'start_sec' (float, taken from the transcript brackets), "
        "'end_sec' (float, start_sec plus the target duration), "
        "'viral_score' (1-100), 'reasoning' (the retention trigger). "
        "You MUST output exactly the requested number of hooks."
    )
    user_prompt = (f"Extract EXACTLY {num_hooks} hooks, each about {target_duration} seconds.\n\n"
                   f"Timestamped transcript:\n{transcript}")
    try:
        reply = _text_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1500)
        hooks = _parse_json(reply)
        if isinstance(hooks, dict):
            hooks = hooks.get("hooks", [])
        total = _get_video_duration(video_file)
        scenes = []
        for h in hooks:
            if "start_sec" not in h:
                continue
            start = max(0.0, float(h["start_sec"]))
            end = min(total, start + target_duration)
            h["start_sec"], h["end_sec"] = round(start, 2), round(end, 2)
            scenes.append({"video_path": str(video_file), "start_sec": h["start_sec"], "end_sec": h["end_sec"]})
        return json.dumps({
            "status": "success",
            "source_video": str(video_file),
            "target_duration_sec": target_duration,
            "hooks": hooks,
            "scenes": scenes,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to extract hooks: {e}"})


@tool
def process_audio_and_captions(video_or_draft_path: str, draft_name: str = "", spoken_query: str = "") -> str:
    """Transcribe a single video file OR all clips of a CapCut draft timeline into an .srt subtitle file."""
    target_path = Path(video_or_draft_path)

    if draft_name and (not target_path.exists() or target_path.is_dir()):
        existing = _find_existing_draft(draft_name)
        if existing:
            target_path = existing
    if not target_path.exists():
        return json.dumps({"error": f"Path not found: {target_path}"})

    model = whisper.load_model("base")

    def _srt_time(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

    def _write_srt(path: Path, segs: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segs, start=1):
                f.write(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text']}\n\n")

    if target_path.is_dir() or target_path.suffix == ".json":
        json_file = target_path / "draft_content.json" if target_path.is_dir() else target_path
        if not json_file.exists():
            return json.dumps({"error": f"draft_content.json not found in {target_path}."})
        try:
            draft_data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            return json.dumps({"error": f"Could not parse draft_content.json: {e}"})

        video_tracks = [t for t in draft_data.get("tracks", []) if t.get("type") == "video"]
        if not video_tracks or not video_tracks[0].get("segments"):
            return json.dumps({"error": "No video clips found on the draft timeline."})

        materials = {m["id"]: m["path"] for m in draft_data.get("materials", {}).get("videos", [])}
        all_segments, offset = [], 0.0
        for seg in video_tracks[0]["segments"]:
            source_file = materials.get(seg.get("material_id"))
            if not source_file or not Path(source_file).exists():
                continue
            src = seg.get("source_timerange", {})
            start_sec = src.get("start", 0) / 1_000_000.0
            dur_sec = src.get("duration", 0) / 1_000_000.0

            temp_wav = Path(tempfile.gettempdir()) / f"temp_seg_{uuid.uuid4().hex[:8]}.wav"
            cmd = ["ffmpeg", "-y", "-ss", str(start_sec)]
            if dur_sec > 0:
                cmd += ["-t", str(dur_sec)]
            cmd += ["-i", str(source_file), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            res = model.transcribe(str(temp_wav), fp16=False)
            for item in res.get("segments", []):
                all_segments.append({"start": item["start"] + offset, "end": item["end"] + offset,
                                     "text": item["text"].strip()})
            offset += dur_sec
            temp_wav.unlink(missing_ok=True)

        out_srt = json_file.parent / f"{json_file.parent.name}_captions.srt"
        _write_srt(out_srt, all_segments)
        return json.dumps({
            "status": "success",
            "srt_file": str(out_srt),
            "transcript_preview": " ".join(s["text"] for s in all_segments)[:300],
            "message": f"Generated subtitles for all {len(video_tracks[0]['segments'])} timeline clips.",
        })

    temp_wav = Path(tempfile.gettempdir()) / f"temp_raw_{uuid.uuid4().hex[:8]}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(target_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    result = model.transcribe(str(temp_wav), fp16=False)
    srt_path = target_path.parent / f"{target_path.stem}_captions.srt"
    _write_srt(srt_path, [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                          for s in result["segments"]])
    temp_wav.unlink(missing_ok=True)
    return json.dumps({"status": "success", "srt_file": str(srt_path),
                       "transcript_preview": result["text"][:300]})


@tool
def detect_and_trim_silence(video_path: str, noise_db: int = -30, min_duration: float = 0.5) -> str:
    """Find silence in a video and return the 'speaking' segments to keep (ready for create_capcut_draft_from_scenes)."""
    target_path = Path(video_path)
    if not target_path.exists():
        return json.dumps({"error": f"File not found: {video_path}"})

    total_duration = _get_video_duration(target_path)
    cmd = ["ffmpeg", "-i", str(target_path), "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
           "-f", "null", "-"]
    try:
        output = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=True).stderr
    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"FFmpeg processing failed: {e}"})

    starts, ends = [], []
    for line in output.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s+([\d\.]+)", line)
            if m: starts.append(float(m.group(1)))
        elif "silence_end" in line:
            m = re.search(r"silence_end:\s+([\d\.]+)", line)
            if m: ends.append(float(m.group(1)))
    if len(starts) > len(ends):
        ends.append(total_duration)

    keep, current = [], 0.0
    for s, e in zip(starts, ends):
        if s > current:
            keep.append({"video_path": str(target_path), "start_sec": round(current, 3), "end_sec": round(s, 3)})
        current = e
    if current < total_duration:
        keep.append({"video_path": str(target_path), "start_sec": round(current, 3),
                     "end_sec": round(total_duration, 3)})

    return json.dumps({"status": "success", "total_silence_clips_removed": len(starts),
                       "keep_segments": keep}, indent=2)


@tool
def find_capcut_drafts_folder() -> str:
    """Locate the CapCut drafts root folder. You normally do NOT need this: draft tools find it themselves."""
    try:
        return json.dumps({"found": True, "drafts_root": str(_drafts_root())})
    except FileNotFoundError as e:
        return json.dumps({"found": False, "message": str(e)})


@tool
def list_capcut_drafts() -> str:
    """List existing CapCut drafts (newest first). Use to find the exact name of a draft the user refers to."""
    try:
        root = _drafts_root()
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    drafts = [d for d in root.iterdir() if d.is_dir() and (d / "draft_content.json").exists()]
    drafts.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return json.dumps({"drafts_root": str(root), "drafts": [d.name for d in drafts[:30]]})


@tool
def create_capcut_draft(clip_paths: list[str], music_path: str = "", is_vertical: bool = False,
                        draft_name: str = "") -> str:
    """Create a NEW CapCut draft from whole clips, with optional background music. Never modifies existing drafts.
    draft_name: short descriptive name (2-4 words, underscores). If empty a name is generated."""
    if not clip_paths:
        return json.dumps({"error": "No clip paths provided"})

    usable, skipped = [], []
    for clip_path in clip_paths:
        p = Path(clip_path)
        if not p.exists():
            skipped.append({"clip": clip_path, "reason": "File not found"})
            continue
        p, note, err = _ensure_capcut_compatible(p)
        if err:
            skipped.append({"clip": clip_path, "reason": err})
            continue
        usable.append((clip_path, p, note))
    if not usable:
        return json.dumps({"error": "No clips could be added to draft", "skipped": skipped})

    try:
        root = _drafts_root()
        name = _resolve_draft_name(root, draft_name, hint=Path(usable[0][0]).stem)
        w, h = (1080, 1920) if is_vertical else (1920, 1080)
        script = cc.DraftFolder(str(root)).create_draft(name, w, h, fps=30, allow_replace=False)
        video_track = script.add_track(cc.TrackType.video)
    except Exception as e:
        return json.dumps({"error": f"Could not create draft: {type(e).__name__}: {e}"})

    current_us, included = 0, []
    for clip_path, p, note in usable:
        try:
            material = cc.VideoMaterial(str(p))
            duration = getattr(material, "duration", 3_000_000)
            video_track.add_segment(cc.VideoSegment(
                material, target_timerange=cc.Timerange(current_us, duration)))
            current_us += duration
            entry = {"clip": clip_path}
            if note: entry["note"] = note
            included.append(entry)
        except Exception as e:
            skipped.append({"clip": clip_path, "reason": str(e)})

    if not included:
        return json.dumps({"error": "No clips could be added to draft", "skipped": skipped})

    if music_path:
        ok, bad = _attach_music(script, music_path, current_us)
        if ok: included.append(ok)
        if bad: skipped.append(bad)

    return _finish_draft(script, root, name, {"included": included, "skipped": skipped})


@tool
def create_capcut_draft_from_scenes(scenes: list[dict], music_path: str = "", is_vertical: bool = False,
                                    draft_name: str = "") -> str:
    """Create a NEW CapCut draft from trimmed scenes [{video_path, start_sec, end_sec}], with optional music.
    Never modifies existing drafts. draft_name: short descriptive name (2-4 words, underscores); if empty one is generated."""
    if not scenes:
        return json.dumps({"error": "No scenes provided"})

    usable, skipped = [], []
    for scene in scenes:
        vp, s, e = scene.get("video_path"), scene.get("start_sec"), scene.get("end_sec")
        if not vp or s is None or e is None or float(e) <= float(s):
            skipped.append({"scene": scene, "reason": "Invalid parameters"})
            continue
        p = Path(vp)
        if not p.exists():
            skipped.append({"scene": scene, "reason": f"File not found: {vp}"})
            continue
        p, note, err = _ensure_capcut_compatible(p)
        if err:
            skipped.append({"scene": scene, "reason": err})
            continue
        usable.append((scene, p, float(s), float(e), note))
    if not usable:
        return json.dumps({"error": "No scenes added to draft", "skipped": skipped})

    try:
        root = _drafts_root()
        name = _resolve_draft_name(root, draft_name, hint=Path(usable[0][0]["video_path"]).stem)
        w, h = (1080, 1920) if is_vertical else (1920, 1080)
        script = cc.DraftFolder(str(root)).create_draft(name, w, h, fps=30, allow_replace=False)
        video_track = script.add_track(cc.TrackType.video)
    except Exception as e:
        return json.dumps({"error": f"Could not create draft: {type(e).__name__}: {e}"})

    current_us, included = 0, []
    for scene, p, start_sec, end_sec, note in usable:
        source_start_us = int(start_sec * 1_000_000)
        duration_us = int((end_sec - start_sec) * 1_000_000)
        try:
            material = cc.VideoMaterial(str(p))
            mat_dur = getattr(material, "duration", None)
            if mat_dur:
                max_end = max(0, mat_dur - 1000)
                if source_start_us >= max_end:
                    skipped.append({"scene": scene, "reason": "Scene starts after clip ends"})
                    continue
                duration_us = min(source_start_us + duration_us, max_end) - source_start_us
            if duration_us <= 0:
                continue
            video_track.add_segment(cc.VideoSegment(
                material,
                target_timerange=cc.Timerange(current_us, duration_us),
                source_timerange=cc.Timerange(source_start_us, duration_us),
            ))
            current_us += duration_us
            included.append(dict(scene, note=note) if note else scene)
        except Exception as e:
            skipped.append({"scene": scene, "reason": str(e)})

    if not included:
        return json.dumps({"error": "No scenes added to draft", "skipped": skipped})

    if music_path:
        ok, bad = _attach_music(script, music_path, current_us)
        if ok: included.append(ok)
        if bad: skipped.append(bad)

    return _finish_draft(script, root, name, {"scenes_included": included, "skipped": skipped})


@tool
def add_music_to_existing_draft(draft_name: str, music_path: str) -> str:
    """Add background music to an EXISTING draft. Only use when the user explicitly asks to modify an existing draft."""
    draft_folder = _find_existing_draft(draft_name)
    if not draft_folder:
        return json.dumps({"error": f"Draft '{draft_name}' not found. Call list_capcut_drafts for exact names."})
    mp = Path(music_path)
    if not mp.exists():
        return json.dumps({"error": f"Music file not found: {music_path}"})

    json_file = draft_folder / "draft_content.json"
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))

        total_us = 0
        for track in data.get("tracks", []):
            if track.get("type") == "video":
                for seg in track.get("segments", []):
                    tr = seg.get("target_timerange", {})
                    total_us = max(total_us, tr.get("start", 0) + tr.get("duration", 0))
        if total_us == 0:
            total_us = 10_000_000

        audio_us = int(_get_video_duration(mp) * 1_000_000)
        target_us = min(audio_us, total_us)
        material_id, track_id, segment_id = (uuid.uuid4().hex for _ in range(3))

        data.setdefault("materials", {}).setdefault("audios", []).append({
            "id": material_id, "path": str(mp.resolve()), "duration": audio_us,
            "type": "audio", "file_Path": str(mp.resolve()), "name": mp.name,
        })
        audio_track = {"id": track_id, "type": "audio", "segments": [{
            "id": segment_id, "material_id": material_id,
            "target_timerange": {"start": 0, "duration": target_us},
            "source_timerange": {"start": 0, "duration": target_us},
            "speed": 1.0, "volume": 1.0,
        }]}
        data["tracks"] = [t for t in data.get("tracks", []) if t.get("type") != "audio"] + [audio_track]

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json.dumps({"status": "success", "draft_name": draft_folder.name, "draft_path": str(draft_folder),
                           "message": f"Added background music {mp.name} to existing draft '{draft_folder.name}'."})
    except Exception as e:
        return json.dumps({"error": f"Failed to add music to existing draft: {e}"})


@tool
def add_scenes_to_existing_draft(scenes: list[dict], draft_name: str) -> str:
    """Append scenes [{video_path, start_sec, end_sec}] to the END of an EXISTING draft. Only when the user explicitly asks."""
    draft_folder = _find_existing_draft(draft_name)
    if not draft_folder:
        return json.dumps({"error": f"Draft '{draft_name}' not found. Call list_capcut_drafts for exact names."})
    if not scenes:
        return json.dumps({"error": "No scenes provided"})

    json_file = draft_folder / "draft_content.json"
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        main_track = next((t for t in data.get("tracks", []) if t.get("type") == "video"), None)
        if not main_track:
            return json.dumps({"error": "No video track found in the existing draft to append to."})

        current_us = 0
        for seg in main_track.get("segments", []):
            tr = seg.get("target_timerange", {})
            current_us = max(current_us, tr.get("start", 0) + tr.get("duration", 0))

        data.setdefault("materials", {}).setdefault("videos", [])
        included, skipped = [], []

        for scene in scenes:
            vp, s, e = scene.get("video_path"), scene.get("start_sec"), scene.get("end_sec")
            if not vp or s is None or e is None:
                skipped.append({"scene": scene, "reason": "Missing required time/path data"})
                continue
            p = Path(vp)
            if not p.exists():
                skipped.append({"scene": scene, "reason": "File not found"})
                continue
            source_start_us = int(float(s) * 1_000_000)
            duration_us = int((float(e) - float(s)) * 1_000_000)
            if duration_us <= 0:
                continue

            material_id, segment_id = uuid.uuid4().hex, uuid.uuid4().hex
            data["materials"]["videos"].append({
                "id": material_id, "path": str(p.resolve()),
                "duration": int(_get_video_duration(p) * 1_000_000),
                "type": "video", "file_Path": str(p.resolve()), "name": p.name,
            })
            main_track["segments"].append({
                "id": segment_id, "material_id": material_id,
                "target_timerange": {"start": current_us, "duration": duration_us},
                "source_timerange": {"start": source_start_us, "duration": duration_us},
                "speed": 1.0, "volume": 1.0,
            })
            current_us += duration_us
            included.append(scene)

        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json.dumps({"status": "success", "draft_name": draft_folder.name,
                           "scenes_added": included, "skipped": skipped})
    except Exception as e:
        return json.dumps({"error": f"Failed to add scenes to existing draft: {e}"})


# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are an interactive, helpful AI post-production coordinator.

If the user mentions a folder or file using casual language (like 'it is on my desktop' or 'in downloads'), DO NOT ask them for the absolute path. Use the locate_user_path tool to find it yourself.

PATH HANDLING: Treat backslashes in Windows paths (C:\\Users\\...) as literal characters, convert them to forward slashes for tool calls, and NEVER ask the user to reformat a path.

When you have the absolute path to a clips folder, ALWAYS call list_video_clips on it first.

# DRAFTS: NAMING AND SAFETY (important)
- Draft tools save into CapCut's drafts folder automatically. NEVER ask the user for a drafts folder and NEVER pass one. You do not choose any paths for drafts.
- When creating a draft, ALWAYS pass a concise, descriptive draft_name based on the user's request (2-4 words, underscores, e.g. 'Arcane_Vertical_Hooks' or 'Silence_Trimmed_Edit'). NEVER ask the user for a name. If the user gave a name, use exactly that.
- Creating a draft ALWAYS makes a brand-new draft. It never changes an existing one. If a name is taken, a numbered name is used automatically.
- Only modify an EXISTING draft when the user explicitly asks to add music/scenes to it (workflows 6 and 7). Use list_capcut_drafts to get its exact name.
- After creating a draft, tell the user the exact draft_name returned by the tool. If the tool result shows verified_on_disk=false or an error, say so honestly. Never claim a draft was created unless the tool returned status 'saved'.
- Call one tool at a time and use the exact values returned by the previous tool.

You have these workflows:

WORKFLOW 0 - Simple draft (DEFAULT when the user just asks to make/build a draft from clips, with no mention of quality, scoring, best clips, searching, hooks, silence or music edits):
1. Call locate_user_path for the folder, then list_video_clips.
2. Immediately call create_capcut_draft with ALL the clips from list_video_clips. Do NOT call evaluate_video_frame.
3. Report the draft_name returned by the tool.

WORKFLOW 1 - General quality curation (ONLY when the user asks to pick, rate, score or filter the best clips):
1. Evaluate every clip using evaluate_video_frame.
2. Include clips with score >= 6.
3. Call create_capcut_draft with the approved clips.
4. Report which clips were included/rejected and the draft name.

WORKFLOW 2 - Content-directed visual scene search:
1. Call find_matching_scenes on EVERY clip in the folder with the user's description.
2. Combine the 'scenes' lists from all clips into one ordered list.
3. Call create_capcut_draft_from_scenes with that combined list.
4. Report which clips contributed scenes and the draft name.

WORKFLOW 3 - Speech-to-text and auto-captions:
1. For subtitles of a draft/timeline, DO NOT ask for video paths. Call list_capcut_drafts if you need the exact draft name.
2. Call process_audio_and_captions with video_or_draft_path set to the folder from find_capcut_drafts_folder and draft_name set to the exact draft name.
3. Tell the user the .srt file was generated and can be dragged onto the CapCut timeline.

WORKFLOW 4 - Viral hooks and vertical social edits (TikTok/Shorts):
1. Call extract_viral_hooks on the video. Pass target_duration if the user gave a length. For time limits (e.g. 'after 1 hour 20 mins') convert to seconds and pass start_time_sec / end_time_sec.
2. Pass the returned 'scenes' list to create_capcut_draft_from_scenes with is_vertical=True.

WORKFLOW 5 - Dead air and silence trimming (jump-cuts):
1. Call detect_and_trim_silence on the video.
2. Pass the returned 'keep_segments' to create_capcut_draft_from_scenes.

WORKFLOW 6 - Add music to an existing draft:
1. Use locate_user_path if the song is in Downloads/Desktop, and list_capcut_drafts for the exact draft name.
2. Call add_music_to_existing_draft with draft_name and music_path.

WORKFLOW 7 - Add scenes to an existing draft:
1. Call find_matching_scenes on the video to locate the footage.
2. Pass the returned 'scenes' to add_scenes_to_existing_draft with the exact draft_name.

BACKGROUND MUSIC DURING CREATION: if the user wants music while creating a draft, find the file with locate_user_path and pass it as music_path to the create tool.

Clips using HEVC/H.265 or 10-bit color are automatically re-encoded to 8-bit H.264; you can mention this briefly if a tool result notes it.
"""


def _clean_agent_output(text: str) -> str:
    cleaned = re.sub(r"<thinking>.*?</thinking>\s*", "", str(text), flags=re.DOTALL)
    cleaned = re.sub(r"</?response>\s*", "", cleaned)
    return cleaned.strip()


agent = Agent(
    model=_build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[
        find_capcut_drafts_folder,
        list_capcut_drafts,
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
    ],
)


def run_chat_loop(agent):
    print("Post-Production Assistant ready. Type your request, or 'exit' to quit.")
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
            print(f"\nAgent: {_clean_agent_output(agent(user_input))}\n")
        except Exception as e:
            print(f"\n[Error during agent turn: {type(e).__name__}: {e}]\n")


def main():
    if len(sys.argv) >= 2:
        clips_folder = Path(sys.argv[1])
        content_query = sys.argv[2] if len(sys.argv) >= 3 else None
        if not clips_folder.exists():
            print(f"Folder not found: {clips_folder}")
            sys.exit(1)
        video_files = [str(p) for p in clips_folder.iterdir()
                       if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
        if not video_files:
            print(f"No video files found in {clips_folder}")
            sys.exit(1)
        print(f"Found {len(video_files)} clip(s): {[Path(v).name for v in video_files]}")
        if content_query:
            prompt = (f"Here are the raw video clips: {video_files}\n"
                      f"Content query: \"{content_query}\"\n"
                      "Use WORKFLOW 2 to search for matching scenes and build the draft.")
        else:
            prompt = (f"Here are the raw video clips: {video_files}\n"
                      "Use WORKFLOW 1 to evaluate quality and build the draft.")
        print("\n--- Agent final response ---")
        print(_clean_agent_output(agent(prompt)))
    else:
        run_chat_loop(agent)


if __name__ == "__main__":
    main()
