"""
Script-To-Video FastAPI background worker.

Accepts a narration script, processes it asynchronously, and returns a finished
MP4 via in-memory job tracking. Heavy work (Gemini, ElevenLabs, Pexels, FFmpeg)
never blocks the POST /api/generate response.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import time
import gc

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError as ElevenLabsApiError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("script-to-video-worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORK_DIR = Path(os.getenv("WORK_DIR", "workspace")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

RAILWAY_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN")

if RAILWAY_DOMAIN:
    # Remove any pre-existing protocol or trailing slashes to normalize
    clean_domain = RAILWAY_DOMAIN.replace("http://", "").replace("https://", "").strip("/")
    # Construct a guaranteed absolute HTTPS address with standard routing
    OUTPUT_BASE_URL = f"https://{clean_domain}/outputs"
else:
    # Fall back to custom variable or localhost if running locally
    OUTPUT_BASE_URL = os.getenv("OUTPUT_BASE_URL", "http://localhost:7860/outputs").rstrip("/")
    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

FFMPEG_BINARY_PATH = "ffmpeg"
FFPROBE_BINARY_PATH = "ffprobe"

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

SUPPORTED_ELEVENLABS_VOICE_PROFILES = (
    "documentary-male",
    "documentary-female",
    "warm-narrator",
)

VOICE_PROFILE_TO_ID = {
    "documentary-male": "pNInz6obpgDQGcFmaJgB",   # True Adam
    "documentary-female": "21m00Tcm4TlvDq8ikWAM", # True Rachel
    "warm-narrator": "cgSgspJ2msm6clMCkdW9",       # True Jessica
}

ELEVENLABS_TTS_MODEL_ID = "eleven_flash_v2_5"  
ELEVENLABS_TTS_OUTPUT_FORMAT = "mp3_22050_32"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

JobStatus = Literal["processing", "completed", "failed"]

# ---------------------------------------------------------------------------
# In-memory job store & persistence helpers
# ---------------------------------------------------------------------------

jobs_db: dict[str, dict[str, Any]] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_job(video_id: str, **fields: Any) -> None:
    """Merge fields into an existing job record and mirror to disk."""
    if video_id not in jobs_db:
        return
    jobs_db[video_id].update(fields)
    jobs_db[video_id]["updated_at"] = utc_now_iso()
    
    # Save a persistent state recovery checkpoint inside the workspace directory
    try:
        job_dir = Path(jobs_db[video_id]["job_dir"])
        if job_dir.exists():
            save_json(job_dir / "job_status.json", jobs_db[video_id])
    except Exception as e:
        logger.warning(f"Failed to persist state checkpoint for job {video_id}: {e}")


# ---------------------------------------------------------------------------
# Script Sanitizer Utility
# ---------------------------------------------------------------------------

def clean_script_text(raw_script: str) -> str:
    """Removes production metadata, timestamps, and formatting cues from the narration."""
    # 1. Drop entire lines that describe visuals or production steps completely
    # This prevents the visual text description from being blended into the voiceover text
    lines = raw_script.splitlines()
    filtered_lines = []
    
    for line in lines:
        # If the line is purely visual or technical notes, skip it completely
        if re.search(r'(?i)^\s*(?:visual|sfx|audio|scene\s*\d*)\b', line):
            continue
        filtered_lines.append(line)
        
    text = "\n".join(filtered_lines)

    # 2. Strip timestamp patterns like '00:00', '1:23', or ranges like '00:02 to 00:10'
    text = re.sub(r'\b\d{1,2}:\d{2}(?:\s*to\s*\d{1,2}:\d{2})?\b', '', text)
    
    # 3. Strip structural layout bracket settings: [Visual: ...] or (SFX: ...)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 4. Clear inline narration prefixes like "Narration:", "Hook:", "Script:"
    text = re.sub(r'(?i)^\s*(?:hook|narration|script|voiceover)\s*:\s*', '', text, flags=re.MULTILINE)
    
    # 5. Normalize whitespace down to clean prose sentences
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=50_000)
    voice_profile: Literal[
        "documentary-male",
        "documentary-female",
        "warm-narrator",
    ] = Field(default="documentary-male")


class GenerateResponse(BaseModel):
    video_id: str
    status: JobStatus


class ScriptSegment(BaseModel):
    segment_id: int
    narration_text: str
    visual_search_query: str


class ParsedScript(BaseModel):
    segments: list[ScriptSegment]


class StatusResponse(BaseModel):
    video_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    step: str = ""
    video_url: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Script-To-Video Worker",
    version="1.0.0",
    description="Background video compilation microservice for faceless automation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(WORK_DIR)), name="outputs")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_video(
    payload: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> GenerateResponse:
    """Accept a script, enqueue background processing, and return immediately."""
    script = payload.script.strip()
    if not script:
        raise HTTPException(status_code=400, detail="Script text is required.")

    video_id = str(uuid.uuid4())
    job_dir = safe_job_dir(video_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    jobs_db[video_id] = {
        "video_id": video_id,
        "status": "processing",
        "progress": 0,
        "step": "Job queued",
        "video_url": None,
        "error": None,
        "voice_profile": payload.voice_profile,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "job_dir": str(job_dir),
    }

    # Save initial configuration payload to disk layout
    save_json(job_dir / "job_status.json", jobs_db[video_id])

    background_tasks.add_task(
        run_video_pipeline,
        video_id=video_id,
        script=script,
        voice_profile=payload.voice_profile,
        job_dir=job_dir,
    )

    logger.info("Queued job %s", video_id)
    return GenerateResponse(video_id=video_id, status="processing")


@app.get("/api/status/{video_id}", response_model=StatusResponse)
async def get_job_status(video_id: str) -> StatusResponse:
    """Return live progress, step message, and final URL when complete."""
    job = jobs_db.get(video_id)
    
    # Context recovery logic if memory was wiped via container replacement cycles
    if not job:
        try:
            fallback_path = WORK_DIR / video_id / "job_status.json"
            if fallback_path.exists():
                job = json.loads(fallback_path.read_text(encoding="utf-8"))
                jobs_db[video_id] = job  # Hydrate cache mapping
                logger.info(f"Successfully re-hydrated job profile {video_id} from storage fallback.")
        except Exception as e:
            logger.error(f"Error reading status fallback for job {video_id}: {e}")

    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{video_id}' not found.")

    return StatusResponse(
        video_id=video_id,
        status=job["status"],
        progress=int(job.get("progress", 0)),
        step=job.get("step", ""),
        video_url=job.get("video_url"),
        error=job.get("error"),
    )


def safe_job_dir(video_id: str) -> Path:
    try:
        uuid.UUID(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid video_id format.") from exc

    job_dir = (WORK_DIR / video_id).resolve()
    if WORK_DIR not in job_dir.parents and job_dir != WORK_DIR:
        raise HTTPException(status_code=400, detail="Unsafe job directory path.")
    return job_dir


def _select_elevenlabs_voice_id(voice_profile: str) -> str:
    return VOICE_PROFILE_TO_ID.get(voice_profile, DEFAULT_VOICE_ID)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
async def run_video_pipeline(
    video_id: str,
    script: str,
    voice_profile: str,
    job_dir: Path,
) -> None:
    try:
        update_job(video_id, step="Sanitizing metadata noise from script...", progress=2)
        clean_script = clean_script_text(script)

        update_job(video_id, step="Parsing script hooks with Gemini...", progress=5)
        segments = await parse_script_with_gemini(clean_script)
        save_json(job_dir / "segments.json", {"segments": [s.model_dump() for s in segments]})

        update_job(video_id, step="Compiling voice over via ElevenLabs...", progress=20)
        voiceover_path = job_dir / "voiceover.mp3"
        await generate_voiceover_elevenlabs(clean_script, voiceover_path, voice_profile)

        update_job(video_id, step="Sourcing b-roll media...", progress=35)
        clip_paths = await run_in_threadpool(
            download_stock_clips,
            segments=segments,
            job_dir=job_dir,
            video_id=video_id
        )

        update_job(video_id, step="Rendering and burn-in subtitles...", progress=65)
        final_local_path = await run_in_threadpool(
            compile_video_with_ffmpeg,
            segments=segments,
            clip_paths=clip_paths,
            voiceover_path=voiceover_path,
            job_dir=job_dir,
            video_id=video_id
        )

        update_job(video_id, step="Uploading to cloud storage...", progress=90)
        public_url = await upload_to_cloud_mock(final_local_path, video_id)

        update_job(
            video_id,
            status="completed",
            progress=100,
            step="Production complete",
            video_url=public_url,
            error=None,
        )
        logger.info("Job %s completed: %s", video_id, public_url)

    except Exception as exc:
        logger.exception("Job %s failed", video_id)
        update_job(
            video_id,
            status="failed",
            step="Production failed",
            error=str(exc),
        )
        

# ---------------------------------------------------------------------------
# Step 1 — Gemini script parsing
# ---------------------------------------------------------------------------

async def parse_script_with_gemini(script: str) -> list[ScriptSegment]:
    USE_MOCK_GEMINI = False  

    if USE_MOCK_GEMINI:
        logger.info("Bypassing Gemini API completely.")
        segments_raw = [
            {
                "segment_id": 1,
                "narration_text": script[:min(len(script), 80)],
                "visual_search_query": "craftsman workshop close up"
            },
            {
                "segment_id": 2,
                "narration_text": script[min(len(script), 80):min(len(script), 160)] or "As we look deeper into the technique.",
                "visual_search_query": "vintage tools woodworking texture"
            }
        ]
    else:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        system_prompt = (
            "You are a documentary video editor. Split the narration script into 4–8 second "
            "segments. Return a JSON object containing a key 'segments' which is an array of objects. "
            "Each object MUST include exactly these keys: 'segment_id' (int starting at 1), "
            "'narration_text' (exact script excerpt), and 'visual_search_query' (3–5 literal concrete "
            "keywords for stock video search — rustic, DIY, heritage crafts style; no abstract phrases)."
        )

        model = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            system_instruction=system_prompt
        )

        try:
            response = model.generate_content(
                script,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.3
                }
            )
            content = response.text
            if not content:
                raise RuntimeError("Gemini returned an empty segmentation response.")
            payload = json.loads(content)
            segments_raw = payload.get("segments")
        except Exception as e:
            logger.warning("Gemini API error. Falling back to mock segments.", e)
            segments_raw = [
                {
                    "segment_id": 1,
                    "narration_text": script[:min(len(script), 80)],
                    "visual_search_query": "craftsman workshop close up"
                },
                {
                    "segment_id": 2,
                    "narration_text": script[min(len(script), 80):min(len(script), 160)] or "As we look deeper into the technique.",
                    "visual_search_query": "vintage tools woodworking texture"
                }
            ]

    if not isinstance(segments_raw, list) or not segments_raw:
        raise RuntimeError("Gemini response missing a non-empty 'segments' array.")

    cleaned_segments = []
    for item in segments_raw:
        if not isinstance(item, dict):
            continue

        if "naration_text" in item and "narration_text" not in item:
            item["narration_text"] = item.pop("naration_text")

        for alternative in ["visual_query", "search_query", "query"]:
            if alternative in item and "visual_search_query" not in item:
                item["visual_search_query"] = item.pop(alternative)

        if isinstance(item.get("visual_search_query"), list):
            item["visual_search_query"] = ", ".join(item["visual_search_query"])
            
        cleaned_segments.append(item)

    return [ScriptSegment.model_validate(item) for item in cleaned_segments]


# ---------------------------------------------------------------------------
# Step 2 — ElevenLabs Voiceover Generation
# ---------------------------------------------------------------------------

async def generate_voiceover_elevenlabs(
    script: str,
    output_path: Path,
    voice_profile: str,
) -> None:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured.")

    def _sync_tts():
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        selected_voice = _select_elevenlabs_voice_id(voice_profile)

        try:
            audio_iterator = client.text_to_speech.convert(
                voice_id=selected_voice,
                text=script,
                model_id=ELEVENLABS_TTS_MODEL_ID,
                output_format=ELEVENLABS_TTS_OUTPUT_FORMAT,
            )

            with open(output_path, "wb") as f:
                for chunk in audio_iterator:
                    if chunk:
                        f.write(chunk)

        except ElevenLabsApiError as exc:
            body = exc.body or {}
            error_detail = body.get("detail") or body.get("message") if isinstance(body, dict) else str(body)
            raise RuntimeError(f"ElevenLabs generation failed: {error_detail}") from exc

    await run_in_threadpool(_sync_tts)


# ---------------------------------------------------------------------------
# Step 3 — Pexels stock video sourcing
# ---------------------------------------------------------------------------

def download_stock_clips(
    segments: list[ScriptSegment],
    job_dir: Path,
    video_id: str,
) -> list[Path]:
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not configured.")

    clips_dir = job_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    total = len(segments)

    for index, segment in enumerate(segments):
        query = segment.visual_search_query.strip()
        progress = 35 + int((index / max(total, 1)) * 25)
        update_job(video_id, step=f"Sourcing b-roll ({index + 1}/{total}): {query[:60]}", progress=progress)

        video_url = search_pexels_best_mp4(query)
        clip_path = clips_dir / f"clip_{segment.segment_id:03d}.mp4"
        download_file(video_url, clip_path)
        downloaded.append(clip_path)

    return downloaded


def search_pexels_best_mp4(query: str) -> str:
    response = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": PEXELS_API_KEY},
        params={"query": query, "per_page": 1, "orientation": "landscape"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    videos = payload.get("videos") or []
    if not videos:
        raise RuntimeError(f"Pexels returned no videos for query: {query}")

    video_files = videos[0].get("video_files") or []
    mp4_files = [f for f in video_files if str(f.get("file_type", "")).lower() == "video/mp4"]
    
    reasonable_files = [
        f for f in mp4_files 
        if (f.get("width") or 0) <= 1920 and (f.get("height") or 0) <= 1080
    ]
    
    target_list = reasonable_files if reasonable_files else mp4_files
    best = max(target_list, key=lambda f: (f.get("height") or 0) * (f.get("width") or 0))
    link = best.get("link")
    if not link:
        raise RuntimeError(f"Pexels MP4 entry missing download link for query: {query}")

    return str(link)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


# ---------------------------------------------------------------------------
# Step 4 — FFmpeg compilation (Fixed Subtitle Speed & Standardization)
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        FFPROBE_BINARY_PATH, "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def compile_video_with_ffmpeg(
    segments: list[ScriptSegment],
    clip_paths: list[Path],
    voiceover_path: Path,
    job_dir: Path,
    video_id: str,
) -> Path:
    if len(segments) != len(clip_paths):
        raise RuntimeError("Segment count does not match downloaded clip count.")

    total_audio_len = get_audio_duration(voiceover_path)
    segment_duration = total_audio_len / len(segments)

    trimmed_clips: list[Path] = []
    for segment, source_clip in zip(segments, clip_paths, strict=True):
        trimmed = job_dir / f"trimmed_{segment.segment_id:03d}.mp4"
        
        # Fixed duration bug: "-stream_loop -1" forces assets to loop seamlessly if shorter than target text window
        subprocess.run(
            [
                FFMPEG_BINARY_PATH, "-y", 
                "-stream_loop", "-1",
                "-i", str(source_clip),
                "-t", f"{segment_duration:.2f}", "-an",
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", str(trimmed)
            ],
            check=True,
            timeout=90
        )
        trimmed_clips.append(trimmed)

    concat_list_path = job_dir / "concat_list.txt"
    lines = []
    for clip in trimmed_clips:
        safe_path = str(clip.resolve()).replace('\\', '/')
        lines.append(f"file '{safe_path}'")
    concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    concatenated_path = job_dir / "concatenated.mp4"
    subprocess.run(
        [
            FFMPEG_BINARY_PATH, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path), "-c", "copy", str(concatenated_path)
        ],
        check=True,
        timeout=90
    )

    with_audio_path = job_dir / "with_audio.mp4"
    subprocess.run(
        [
            FFMPEG_BINARY_PATH, "-y", "-i", str(concatenated_path), "-i", str(voiceover_path),
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(with_audio_path)
        ],
        check=True,
        timeout=90
    )

    subtitles_path = job_dir / "subtitles.srt"
    cursor = 0.0
    entries = []
    for segment in segments:
        start = cursor
        end = cursor + segment_duration
        cursor = end
        entries.append(f"{segment.segment_id}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{segment.narration_text.strip()}\n")
    subtitles_path.write_text("\n".join(entries), encoding="utf-8")

    final_path = (WORK_DIR / video_id / "final.mp4").resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    
    # OS-Safe filter formatting to accommodate Windows local tests and Railway Linux environments smoothly
    if sys.platform == "win32":
        safe_sub_path = str(subtitles_path.resolve()).replace('\\', '/').replace(':', '\\:')
    else:
        safe_sub_path = str(subtitles_path.resolve())
        
    sub_filter = f"subtitles='{safe_sub_path}'"
    
    subprocess.run(
        [
            FFMPEG_BINARY_PATH, "-y", 
            "-i", str(with_audio_path), 
            "-vf", sub_filter, 
            "-c:v", "libx264", "-preset", "ultrafast", 
            "-c:a", "copy", 
            str(final_path)
        ],
        check=True,
        timeout=300
    )

    time.sleep(1.0)
    gc.collect()
    return final_path


def format_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Step 5 — Asset monitoring & delivery verification
# ---------------------------------------------------------------------------

async def upload_to_cloud_mock(local_path: Path, video_id: str) -> str:
    """
    Bypasses cloud providers by validating that the file was generated 
    directly in the public assets directory, avoiding Windows file locks.
    """
    logger.info(f"Verifying final asset handle stability for ID: {video_id}...")
    
    last_size = -1
    for _ in range(12):  # Up to 6 seconds of stream stabilization checks
        if local_path.exists():
            current_size = local_path.stat().st_size
            if current_size > 0 and current_size == last_size:
                break  
            last_size = current_size
        await run_in_threadpool(time.sleep, 0.5)

    if not local_path.exists() or local_path.stat().st_size == 0:
        raise RuntimeError(f"Final render at {local_path} is missing or empty (0 bytes).")

    logger.info(f"Asset verified at static distribution layout. Size: {local_path.stat().st_size} bytes.")
    return f"{OUTPUT_BASE_URL}/{video_id}/final.mp4"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)