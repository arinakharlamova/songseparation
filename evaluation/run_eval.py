"""
Оценка качества модели Demucs htdemucs на датасете MUSDB18-7.
Рассчитывает метрики SDR, SIR, SAR с помощью museval.
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import museval
import musdb
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Добавляем корневую директорию проекта в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.separator import DemucsSeparator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Путь к датасету MUSDB18-7 (настроен под локальное окружение)
DATASET_ROOT = "C:/Users/Arina/MUSDB18/MUSDB18-7"
RESULTS_DIR = Path("evaluation_results")
NUM_TRACKS = 5  # Количество треков для оценки
STEMS = ["vocals", "drums", "bass", "other"]


def evaluate_track(separator: DemucsSeparator, track, track_idx: int, total: int) -> dict:
    """Разделяет трек на стемы и вычисляет метрики качества."""
    logger.info(f"[{track_idx+1}/{total}] Оценка: {track.name}")

    sr = int(track.rate)

    # Сохраняем микс во временный WAV через soundfile (обход проблемы с torchcodec на Windows)
    tmp_path = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_path, track.audio, sr)

    try:
        start = time.time()
        result = separator.separate(tmp_path)
        sep_time = time.time() - start

        # Формируем 3D-массивы для museval: (nsrc, nsampl, nchan)
        nsrc = len(STEMS)
        # Находим общую длину всех стемов
        min_len = None
        for s in STEMS:
            est = result.stems[s]
            gt = track.targets[s].audio.astype(np.float32)  # (samples, channels)
            if est.ndim == 1:
                est = est[np.newaxis, :]
            est_t = est.T  # (samples, channels)
            l = min(est_t.shape[0], gt.shape[0])
            if min_len is None:
                min_len = l
            else:
                min_len = min(min_len, l)

        est_arr = np.zeros((nsrc, min_len, 2), dtype=np.float32)
        gt_arr = np.zeros((nsrc, min_len, 2), dtype=np.float32)

        # Заполняем массивы оценёнными и эталонными стемами
        for i, s in enumerate(STEMS):
            est = result.stems[s]
            gt = track.targets[s].audio.astype(np.float32)
            if est.ndim == 1:
                est = est[np.newaxis, :]
            est_t = est.T  # (samples, channels)
            l = min(est_t.shape[0], gt.shape[0])
            est_arr[i, :l, :] = est_t[:l, :]
            gt_arr[i, :l, :] = gt[:l, :]

        # Вычисляем метрики через museval (ожидает 3D-массивы: nsrc, nsampl, nchan)
        sdr, isr, sir, sar = museval.evaluate(
            gt_arr,
            est_arr,
            win=sr,
            hop=sr,
            mode="v4",
        )

        # sdr, sir, sar — массивы формы (nsrc, n_windows), берём медиану по окнам
        metrics = {}
        for i, s in enumerate(STEMS):
            metrics[s] = {
                "SDR": float(np.median(sdr[i][~np.isnan(sdr[i])])),
                "SIR": float(np.median(sir[i][~np.isnan(sir[i])])),
                "SAR": float(np.median(sar[i][~np.isnan(sar[i])])),
            }

        logger.info(f"  Готово за {sep_time:.1f}с")
        for s in STEMS:
            logger.info(f"  {s}: SDR={metrics[s]['SDR']:.2f}, SIR={metrics[s]['SIR']:.2f}, SAR={metrics[s]['SAR']:.2f}")

        return {
            "track": track.name,
            "time": sep_time,
            "metrics": metrics,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Ошибка на треке {track.name}: {e}")
        import traceback
        traceback.print_exc()
        return {"track": track.name, "error": str(e), "success": False}
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    """Запуск оценки модели на датасете MUSDB18-7."""
    logger.info(f"Загрузка Demucs htdemucs на {'cuda' if torch.cuda.is_available() else 'cpu'}")
    separator = DemucsSeparator(
        model_name="htdemucs",
        device=None,  # автоопределение
        chunk_size=None,  # обрабатываем трек целиком для оценки
    )

    mus = musdb.DB(root=DATASET_ROOT, subsets="test")
    tracks = list(mus.tracks)[:NUM_TRACKS]
    logger.info(f"Оценка {len(tracks)} треков на MUSDB18-7")

    results = []
    for i, track in enumerate(tracks):
        r = evaluate_track(separator, track, i, len(tracks))
        results.append(r)

    # Сохраняем сырые результаты
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump({"model": "htdemucs", "tracks": results}, f, indent=2)

    # Формируем сводную таблицу
    rows = []
    for r in results:
        if not r.get("success"):
            continue
        for s in STEMS:
            if s in r.get("metrics", {}):
                rows.append({
                    "track": r["track"],
                    "stem": s,
                    "SDR": r["metrics"][s]["SDR"],
                    "SIR": r["metrics"][s]["SIR"],
                    "SAR": r["metrics"][s]["SAR"],
                    "time_s": r["time"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    logger.info(f"\nРезультаты сохранены в {RESULTS_DIR}")

    # Вывод сводки
    print("\n" + "=" * 80)
    print("ИТОГИ ОЦЕНКИ — Demucs htdemucs на MUSDB18-7")
    print("=" * 80)
    print(f"Оценено треков: {len(results)}")
    print(f"Устройство: {separator.device}")
    print()

    if df.empty:
        print("Нет успешных оценок.")
        return

    # Средние значения по стемам
    print(f"{'Стем':<10} {'SDR':>8} {'SIR':>8} {'SAR':>8}")
    print("-" * 40)
    for s in STEMS:
        sdf = df[df["stem"] == s]
        if len(sdf) > 0:
            print(f"{s:<10} {sdf['SDR'].mean():>8.2f} {sdf['SIR'].mean():>8.2f} {sdf['SAR'].mean():>8.2f}")
    print()

    # Общие средние значения
    print(f"{'Общее':<10} {df['SDR'].mean():>8.2f} {df['SIR'].mean():>8.2f} {df['SAR'].mean():>8.2f}")
    print()

    # Детализация по трекам
    print("-" * 80)
    for r in results:
        if r.get("success"):
            print(f"{r['track']}")
            for s in STEMS:
                m = r["metrics"][s]
                print(f"  {s:>10}: SDR={m['SDR']:+.2f}  SIR={m['SIR']:+.2f}  SAR={m['SAR']:+.2f}")
            print(f"  {'время':>10}: {r['time']:.1f}с")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()
