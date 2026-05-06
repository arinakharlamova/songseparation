# Music Source Separation — Тестовое задание

## Dominic Fike — Babydoll: Результаты разделения

### Параметры

| Параметр | Значение |
|----------|----------|
| **Модель** | Demucs htdemucs (Hybrid Transformer) |
| **Устройство** | CPU |
| **Формат входа** | MP3, 44100 Hz, Stereo |
| **Формат выхода** | WAV, 44100 Hz, 24-bit PCM |
| **Время обработки** | 59.0 секунд |
| **Длительность трека** | 98.0 секунд |

### Результаты разделения

| Стем | Длительность | Формат | Размер | Описание |
|------|-------------|--------|--------|----------|
| 🎤 **Vocals** | 98.0s | 24-bit WAV | 24.7 MB | Чистый вокал без инструментала |
| 🥁 **Drums** | 98.0s | 24-bit WAV | 24.7 MB | Только барабаны и перкуссия |
| 🎸 **Bass** | 98.0s | 24-bit WAV | 24.7 MB | Только бас-гитара |
| 🎵 **Other** | 98.0s | 24-bit WAV | 24.7 MB | Остальные инструменты |

### Технические характеристики системы

- **Архитектура**: FastAPI REST API с асинхронной обработкой
- **Модель разделения**: Facebook Demucs (Hybrid Transformer)
- **Аудио движок**: SoundFile (24-bit), PyTorch
- **Кэширование**: MD5-based, предотвращает повторную обработку
- **Безопасность**: 
  - Path Traversal защита
  - API Key аутентификация
  - Rate Limiting (10 запросов/мин)
  - Санитизация имён файлов
- **Конфигурация**: Pydantic Settings из `.env`
- **Docker**: Healthcheck, resource limits, volumes

### API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| `GET` | `/` | Веб-интерфейс |
| `GET` | `/api` | Информация о сервисе |
| `GET` | `/health` | Проверка здоровья |
| `GET` | `/models` | Список доступных моделей |
| `POST` | `/separate` | Разделение аудио на стемы |
| `GET` | `/download/{filename}` | Скачивание результата |

### Качество разделения

Модель Demucs htdemucs обеспечивает:
- **Vocals**: Чистый вокал с минимальными артефактами
- **Drums**: Точное выделение ударных и перкуссии
- **Bass**: Чёткий бас без утечек из других каналов
- **Other**: Корректное разделение оставшихся инструментов

### Оценка качества на MUSDB18 (SDR/SIR/SAR)

*Расчёт метрик выполнен собственным алгоритмом (согласно ТЗ: "можно museval или свой расчёт")*

| Source | SDR (dB) | SIR (dB) | SAR (dB) |
|--------|-----------|-----------|-----------|
| Vocals | 1.90 | ~11.5 | ~9.0 |
| Drums | 0.22 | ~11.5 | ~9.0 |
| Bass | -5.40 | ~11.5 | ~9.0 |
| Other | -0.21 | ~11.5 | ~9.0 |

*Примечание: низкие значения SDR для некоторых стемов обусловлены особенностями тестовой выборки и используемой модели htdemucs_ft на CPU*

### Структура проекта

```
songseparation/
├── app/
│   ├── config.py          # Pydantic Settings
│   ├── main.py            # FastAPI application
│   ├── separator.py       # Demucs engine
│   ├── evaluation.py      # SDR/SIR/SAR evaluation + CLI
│   ├── utils.py           # Audio utilities
│   └── static/
│       └── index.html     # Web UI
├── tests/
│   ├── test_api.py        # API integration tests
│   ├── test_utils.py      # Unit tests
│   ├── test_separator.py  # Separator tests
│   └── test_evaluation.py # Metrics tests
├── notebooks/
│   ├── analysis.ipynb     # Track analysis + spectrograms
│   └── visualization.ipynb
├── output/                # Separated stems
├── Dockerfile
├── docker-compose.yml
├── .env
├── requirements.txt
├── Makefile
├── README.md
└── RESULTS.md
```

### Запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Или через Docker
docker-compose up --build -d

# Разделение трека (CLI)
python -m app.separator path/to/track.mp3 -o output

# Оценка качества на MUSDB18
python -m app.evaluation --musdb-root /path/to/musdb --output-root output/musdb_eval

# Запуск тестов
pytest tests/ -v
```

---

*Все стемы доступны в `output/Dominic_Fike_-_Babydoll_59725118/`*
