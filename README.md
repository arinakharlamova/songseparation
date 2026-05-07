# Сервис разделения музыки на стемы

Прототип сервиса для разделения музыки на стемы (вокал, барабаны, бас, остальное), аналогичного Moises.

## Стек

- **Модель**: Demucs htdemucs (Meta) — гибридный трансформер, лучший баланс качество/скорость
- **Бэкенд**: FastAPI
- **Обработка аудио**: torchaudio, soundfile, pydub
- **Оценка качества**: museval (SDR, SIR, SAR) на MUSDB18-HQ
- **Контейнеризация**: Docker + docker-compose

## Быстрый запуск

### Локально

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Откройте http://localhost:8000 для веб-интерфейса,
http://localhost:8000/docs для документации API.

### Docker

```bash
docker-compose up --build
```

## API

### POST /separate

Загрузите аудиофайл для разделения.

```bash
curl -X POST http://localhost:8000/separate \
  -F "file=@track.mp3"
```

**Ответ:**
```json
{
  "id": "abc123",
  "filename": "track.mp3",
  "stems": {
    "vocals": "http://localhost:8000/download/abc123/vocals.wav",
    "drums": "http://localhost:8000/download/abc123/drums.wav",
    "bass": "http://localhost:8000/download/abc123/bass.wav",
    "other": "http://localhost:8000/download/abc123/other.wav"
  },
  "processing_time": 6.5,
  "sample_rate": 44100,
  "model": "htdemucs",
  "device": "cpu"
}
```

### GET /health

Проверка готовности сервиса.

### GET /download/{job_id}/{stem_name}

Скачать отдельный стем в формате WAV.

### POST /separate?return_zip=true

Возвращает ZIP-архив со всеми стемами.

## Оценка качества на MUSDB18-7

5 тестовых треков оценены с помощью museval (метрики SDR, SIR, SAR):

| Стем   | SDR (dB) | SIR (dB) | SAR (dB) |
|--------|----------|----------|----------|
| Vocals | 9.28     | 15.36    | 10.21    |
| Drums  | 5.71     | 8.58     | 8.44     |
| Bass   | 9.83     | 16.33    | 11.06    |
| Other  | 3.90     | 6.23     | 4.84     |
| **Общее** | **7.18** | **11.62** | **8.64** |

Запуск оценки:
```bash
python evaluation/run_eval.py
```

Просмотр интерактивного анализа:
```bash
jupyter notebook evaluation/metrics.ipynb
```

## Бенчмарки

Тестовое железо:
- **CPU**: Intel Core i5-1135G7 @ 2.40 GHz (4 ядра / 8 потоков)
- **RAM**: 16 GB
- **GPU**: отсутствует (Intel Iris Xe Graphics — не поддерживается CUDA)
- **OS**: Windows 11, Python 3.13
- **Модель**: Demucs htdemucs (PyTorch CPU)

| Трек | Время обработки | Коэфф. реального времени |
|------|-----------------|--------------------------|
| 6.8 с (MUSDB18-7) | ~6.5 с | ~1.0x |
| 5 мин (полный трек) | ~4-5 мин | ~1.0x |

На NVIDIA GPU (например, RTX 3060) ожидается ускорение 10-20x:
- Трек 6.8 с: ~0.3-0.7 с
- Трек 5 мин: ~20-30 с

Время обработки растёт примерно линейно с длительностью трека.

## Оптимизации

- **Обрезка тишины**: Удаление тишины в начале и конце трека
- **Чанковая обработка**: Длинные треки разбиваются на сегменты по 30 с с перекрытием 1 с и кроссфейд-окном
- **Кэширование**: Результаты кэшируются по SHA-256 хешу файла (массивы numpy)
- **Поддержка GPU**: Автоопределение CUDA, настраивается через `MUSIC_SEP_DEVICE=cuda`
- **Таймаут запросов**: Защита от зависания через `asyncio.wait_for`
- **Таймаут запроса**: Настраивается через MUSIC_SEP_REQUEST_TIMEOUT (по умолчанию 60 с)

## Структура проекта

```
separationtz/
├── app/
│   ├── main.py           # FastAPI приложение
│   ├── separator.py      # Класс DemucsSeparator
│   ├── config.py         # Настройки через Pydantic
│   └── utils.py          # Валидация аудио, создание ZIP
├── evaluation/
│   ├── run_eval.py       # Скрипт оценки на MUSDB18
│   └── metrics.ipynb     # Jupyter ноутбук с анализом
├── frontend/
│   └── index.html        # Веб-интерфейс (drag & drop, плееры)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Выбор модели

Модель Demucs htdemucs выбрана по причинам:
- Лучший SDR среди открытых моделей (по сравнению со Spleeter, Open-Unmix)
- Гибридная трансформерная архитектура хорошо обрабатывает сложные смеси
- Встроенная чанковая обработка и перекрытие для треков произвольной длины
- Активная поддержка командой FAIR из Meta

## Известные ограничения

- MUSDB18-7 использует 6.8-секундные отрывки, а не полные треки — метрики могут отличаться на целых песнях
- Инференс на CPU медленный для длинных треков (~1x реального времени); рекомендуется GPU
- Стем «Остальное» стабильно показывает худшие результаты (SDR ~4 дБ), так как содержит разнородную смесь оставшихся инструментов
- Нет очереди асинхронной обработки — запросы блокируются на время разделения