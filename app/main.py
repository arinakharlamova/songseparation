"""
FastAPI приложение для сервиса разделения музыкальных источников.
"""

import asyncio
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
import uvicorn

from app.separator import DemucsSeparator
from app.config import settings
from app.utils import (
    validate_audio_file,
    create_stem_zip,
    ensure_directory,
    cleanup_temp_files,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# Глобальный экземпляр разделителя
separator: Optional[DemucsSeparator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения (запуск/завершение)."""
    global separator
    
    # Запуск
    logger.info("Запуск сервиса разделения музыки...")
    
    # Создание необходимых директорий
    for dir_path in [settings.cache_dir, settings.output_dir, settings.temp_dir]:
        ensure_directory(dir_path)
    
    # Инициализация разделителя
    try:
        separator = DemucsSeparator(
            model_name=settings.model_name,
            device=settings.device,
            cache_dir=settings.cache_dir,
            chunk_size=settings.chunk_size,
            overlap=settings.overlap,
            trim_silence=settings.trim_silence,
            silence_threshold=settings.silence_threshold,
        )
        logger.info("Разделитель успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации разделителя: {e}")
        raise
    
    # Очистка старых временных файлов
    cleanup_temp_files(settings.temp_dir)
    
    yield
    
    # Завершение
    logger.info("Завершение работы сервиса разделения музыки...")
    if separator:
        separator.clear_cache()
    
    cleanup_temp_files(settings.temp_dir, max_age_hours=0)  # Удалить все


# Инициализация FastAPI приложения
app = FastAPI(
    title="API разделения музыкальных источников",
    description="Разделение музыки на вокал, барабаны, бас и остальные инструменты с помощью Demucs",
    version="1.0.0",
    lifespan=lifespan,
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: логирование времени запросов + таймаут обработки
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware: логирует время запроса и ограничивает время обработки."""
    start_time = time.time()

    try:
        # Применяем таймаут ко всем запросам
        response = await asyncio.wait_for(
            call_next(request),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"Таймаут запроса через {settings.request_timeout}с: {request.url.path}")
        return JSONResponse(
            status_code=504,
            content={"detail": f"Превышено время обработки ({settings.request_timeout} секунд)"},
        )

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Статус: {response.status_code} - "
        f"Время: {process_time:.3f}с"
    )

    return response


@app.get("/")
async def root():
    """Главная страница — веб-интерфейс."""
    from fastapi.responses import HTMLResponse
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return {"message": "Frontend not found", "docs": "/docs"}


@app.get("/api/info")
async def api_info():
    """Информация о сервисе."""
    return {
        "service": "Music Source Separation API",
        "version": "1.0.0",
        "model": settings.model_name,
        "device": settings.device or ("cuda" if __import__('torch').cuda.is_available() else "cpu"),
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    global separator
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Разделитель не инициализирован")
    
    import torch
    
    return {
        "статус": "работает",
        "модель_загружена": separator.model is not None,
        "устройство": separator.device,
        "cuda_доступна": torch.cuda.is_available(),
        "размер_кэша": len(list(separator.cache_dir.glob("*.npy"))) if separator.cache_dir else 0,
    }


@app.post("/separate")
async def separate_audio(
    request: Request,
    file: UploadFile = File(...),
    return_zip: bool = False,
):
    """
    Разделение аудиофайла на стемы.
    
    Аргументы:
        file: Аудиофайл (WAV, MP3, FLAC, OGG, M4A)
        return_zip: Если True, возвращает ZIP архив со всеми стемами
        
    Возвращает:
        JSON с URL стемов или ZIP файл
        
    Исключения:
        HTTPException: При ошибках валидации или обработки
    """
    global separator
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    
    # Валидация расширения файла
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {file_extension}. Поддерживаемые: {supported_extensions}"
        )
    
    # Сохранение загруженного файла во временную директорию
    temp_id = uuid.uuid4().hex
    temp_path = settings.temp_dir / f"{temp_id}{file_extension}"
    
    try:
        # Сохранение файла с проверкой размера
        with open(temp_path, "wb") as buffer:
            file_size = 0
            while chunk := await file.read(8192):  # Чтение чанками
                file_size += len(chunk)
                if file_size > settings.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл слишком большой. Максимальный размер: {settings.max_file_size // (1024*1024)}МБ"
                    )
                buffer.write(chunk)
        
        logger.info(f"Получен файл: {file.filename} ({file_size} байт)")
        
        # Валидация файла
        is_valid, error_msg = validate_audio_file(
            temp_path,
            settings.max_file_size,
            settings.max_duration,
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Обработка с таймаутом
        try:
            # Выполнение разделения
            result = separator.separate(temp_path)
            
            # Сохранение стемов в файлы
            output_dir = settings.output_dir / temp_id
            stem_paths = separator.separate_to_files(temp_path, output_dir)
            
            # Подготовка ответа
            base_url = str(request.base_url).rstrip('/')
            
            if return_zip:
                # Создание ZIP архива с русскими названиями
                zip_buffer = create_stem_zip(stem_paths)
                
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename=stems_{temp_id}.zip"
                    }
                )
            else:
                # Возврат JSON с URL
                stem_urls = {
                    stem: f"{base_url}/download/{temp_id}/{stem}.wav"
                    for stem in stem_paths.keys()
                }
                
                # Русские названия стемов для ответа
                stems_ru = {
                    'vocals': 'вокал',
                    'drums': 'барабаны', 
                    'bass': 'бас',
                    'other': 'остальное'
                }
                
                return JSONResponse({
                    "id": temp_id,
                    "имя_файла": file.filename,
                    "стемы": {stems_ru.get(k, k): v for k, v in stem_urls.items()},
                    "время_обработки": result.processing_time,
                    "частота_дискретизации": result.sample_rate,
                    "модель": result.model_name,
                    "устройство": result.device,
                })
                
        except Exception as e:
            logger.error(f"Ошибка разделения: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    finally:
        # Очистка временного файла
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as e:
                logger.warning(f"Не удалось очистить временный файл {temp_path}: {e}")


@app.get("/download/{job_id}/{stem_name}")
async def download_stem(job_id: str, stem_name: str):
    """
    Скачивание конкретного стема.
    
    Аргументы:
        job_id: Идентификатор задачи
        stem_name: Имя файла стема
        
    Возвращает:
        WAV файл
    """
    stem_path = settings.output_dir / job_id / stem_name
    
    if not stem_path.exists():
        raise HTTPException(status_code=404, detail="Стем не найден")
    
    return FileResponse(
        stem_path,
        media_type="audio/wav",
        filename=stem_name,
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Удаление файлов задачи.
    
    Аргументы:
        job_id: Идентификатор задачи
        
    Возвращает:
        Подтверждение удаления
    """
    job_dir = settings.output_dir / job_id
    
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    try:
        shutil.rmtree(job_dir)
        return {"сообщение": f"Задача {job_id} успешно удалена"}
    except Exception as e:
        logger.error(f"Ошибка удаления задачи {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка удаления задачи")


@app.get("/cache/stats")
async def cache_stats():
    """Получение статистики кэша."""
    global separator
    
    if separator is None or separator.cache_dir is None:
        return {"кэширование": False, "файлов": 0, "размер_мб": 0}
    
    cache_files = list(separator.cache_dir.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "кэширование": True,
        "файлов": len(cache_files),
        "размер_мб": total_size / (1024 * 1024),
        "директория_кэша": str(separator.cache_dir),
    }


@app.delete("/cache/clear")
async def clear_cache():
    """Очистка всех кэшированных результатов."""
    global separator
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    
    count = separator.clear_cache()
    return {"сообщение": f"Очищено {count} файлов кэша"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        reload=settings.debug,
    )