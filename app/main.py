"""
FastAPI application for source separation service.
Production-ready with security, validation, and observability.
"""
import gc
import re
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
import fastapi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.separator import SourceSeparator
from app.utils import compute_file_hash, create_stem_archive, validate_audio_format

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    "logs/api.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# ---------------------------------------------------------------------------
# Pydantic models (OpenAPI schemas)
# ---------------------------------------------------------------------------
class SeparationResponse(BaseModel):
    task_id: str
    status: str
    archive_url: Optional[str] = None
    stems: Optional[dict[str, str]] = None
    duration: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class ModelInfo(BaseModel):
    name: str
    description: str


class ModelsResponse(BaseModel):
    current: str
    available: list[ModelInfo]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if settings.api_key:
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory)
# ---------------------------------------------------------------------------
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = 60.0
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < window
    ]
    if len(_rate_limit_store[client_ip]) >= settings.rate_limit_per_minute:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w.\-]", "_", name)
    name = name.lstrip(".")
    return name or "unnamed"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    logger.info(f"Device: {settings.resolved_device}")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Max file size: {settings.max_file_size_mb}MB")
    logger.info(f"Max duration: {settings.max_duration_seconds}s")

    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    _cleanup_old_cache()

    yield

    logger.info("API shutting down...")
    _cleanup_temp()
    if separator_instance[0] is not None:
        separator_instance[0] = None
        gc.collect()


def _cleanup_temp():
    temp = Path(settings.temp_dir)
    if temp.exists():
        for f in temp.iterdir():
            try:
                f.unlink()
            except Exception:
                pass


def _cleanup_old_cache():
    cache = Path(settings.cache_dir)
    if not cache.exists():
        return
    cutoff = time.time() - (settings.cache_cleanup_hours * 3600)
    for d in cache.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Cleaned old cache: {d.name}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Source Separation API",
    description="Music source separation service using Demucs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

separator_instance: list[Optional[SourceSeparator]] = [None]


def get_separator() -> SourceSeparator:
    if separator_instance[0] is None:
        logger.info("Initializing separator...")
        separator_instance[0] = SourceSeparator(
            model_name=settings.model_name,
            device=settings.resolved_device,
            output_dir=settings.output_dir,
        )
    return separator_instance[0]


# ---------------------------------------------------------------------------
# Middleware: client IP for rate limiting
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if request.method == "POST" and not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": f"Max {settings.rate_limit_per_minute} requests per minute"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root_ui():
    """Serve web UI."""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api", response_model=dict)
async def root():
    return {
        "service": "Source Separation API",
        "version": "1.0.0",
        "model": settings.model_name,
        "device": settings.resolved_device,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "device": settings.resolved_device,
        "model_loaded": separator_instance[0] is not None,
    }


@app.post("/separate", response_model=SeparationResponse)
async def separate_audio(
    file: UploadFile = File(...),
    return_zip: bool = Form(True),
    _auth: bool = Depends(verify_api_key),
):
    task_id = str(uuid.uuid4())[:8]

    safe_name = sanitize_filename(file.filename or "upload")
    file_ext = Path(safe_name).suffix.lower()

    if file_ext not in settings.ext_set:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(sorted(settings.ext_set))}",
        )

    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_path = temp_dir / f"{task_id}{file_ext}"

    try:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {settings.max_file_size_mb}MB",
            )

        async with aiofiles.open(input_path, "wb") as f:
            await f.write(content)

        is_valid, info = validate_audio_format(input_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=info)

        logger.info(f"[{task_id}] File valid: {info}")

        # Duration check
        sf_info = sf.info(str(input_path))
        duration = sf_info.duration

        if duration > settings.max_duration_seconds:
            raise HTTPException(
                status_code=400,
                detail=f"File too long. Max: {settings.max_duration_seconds}s",
            )

        logger.info(f"[{task_id}] Duration: {duration:.1f}s")

        # Cache check
        file_hash = compute_file_hash(input_path)
        cache_dir = Path(settings.cache_dir) / file_hash

        if cache_dir.exists() and list(cache_dir.glob("*.wav")):
            logger.info(f"[{task_id}] Cache hit: {file_hash[:8]}")
            stems_dir = cache_dir
        else:
            logger.info(f"[{task_id}] Starting separation...")
            sep = get_separator()

            try:
                stems_paths = sep.separate_with_timeout(
                    input_path,
                    timeout_seconds=settings.timeout_seconds,
                )
                stems_dir = stems_paths[list(stems_paths.keys())[0]].parent

                cache_dir.mkdir(parents=True, exist_ok=True)
                for src in stems_paths.values():
                    shutil.copy2(src, cache_dir / src.name)

            except TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail="Separation timeout. Try shorter file.",
                )
            except Exception as e:
                logger.error(f"[{task_id}] Separation error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Separation failed: {str(e)}",
                )

        if return_zip:
            archive_path = create_stem_archive(stems_dir)
            return SeparationResponse(
                task_id=task_id,
                status="completed",
                archive_url=f"/download/{archive_path.name}",
                duration=duration,
            )
        else:
            stems_map = {}
            for stem_path in stems_dir.glob("*.wav"):
                stems_map[stem_path.stem] = f"/download/{stem_path.name}"
            return SeparationResponse(
                task_id=task_id,
                status="completed",
                stems=stems_map,
                duration=duration,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{task_id}] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


@app.get("/download/{filename}")
async def download_file(filename: str):
    safe = sanitize_filename(filename)

    # Only allow safe filenames (no path components)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    output_root = Path(settings.output_dir).resolve()

    search_paths = [
        output_root / safe,
        Path(settings.cache_dir).resolve() / safe,
        Path(settings.temp_dir).resolve() / safe,
    ]

    for path in search_paths:
        # Ensure resolved path is inside output root
        resolved = path.resolve()
        if not str(resolved).startswith(str(output_root)):
            raise HTTPException(status_code=403, detail="Access denied")
        if resolved.exists() and resolved.is_file():
            return FileResponse(
                path=str(resolved),
                filename=safe,
                media_type="application/octet-stream",
            )

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(
        current=settings.model_name,
        available=[
            ModelInfo(name="htdemucs_ft", description="Best quality, slowest"),
            ModelInfo(name="htdemucs", description="Good quality, faster"),
            ModelInfo(name="demucs", description="Standard quality, fastest"),
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
