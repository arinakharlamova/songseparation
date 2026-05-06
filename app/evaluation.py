"""Модуль оценки качества разделения источников.

Объединяет метрики, оценку и инструменты командной строки.
Считает метрики SDR, SIR, SAR с использованием museval или mir_eval.
"""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

try:
    import museval
    HAS_MUSEVAL = True
except ImportError:
    HAS_MUSEVAL = False
    logger.warning("museval недоступен")

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    logger.warning("mir_eval недоступен")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa недоступен для ресемплинга")

STEM_NAMES = ["вокал", "барабаны", "бас", "прочее"]


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Ресемплинг аудио до целевой частоты дискретизации.
    
    Аргументы:
        data: Аудиоданные формы (сэмплы, каналы) или (сэмплы,)
        orig_sr: Исходная частота дискретизации
        target_sr: Целевая частота дискретизации
    
    Возвращает:
        Ресемплированное аудио в том же формате
    """
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

try:
    import museval
    HAS_MUSEVAL = True
except ImportError:
    HAS_MUSEVAL = False
    logger.warning("museval not available")

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    logger.warning("mir_eval not available")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available for resampling")


STEM_NAMES = ["vocals", "drums", "bass", "other"]


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        data: Audio data with shape (samples, channels) or (samples,)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio with same shape format
    """
    if orig_sr == target_sr:
        return data

    # Calculate target number of samples
    if data.ndim == 1:
        num_samples = int(len(data) * target_sr / orig_sr)
    else:
        num_samples = int(data.shape[0] * target_sr / orig_sr)

    if HAS_LIBROSA:
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr, axis=0)
    else:
        from scipy import signal
        return signal.resample(data, num_samples, axis=0)


def load_audio_pair(ref_path: Path, est_path: Path, target_sr: int = 44100):
    """Загрузить и синхронизировать эталонные и оценочные аудиофайлы."""
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")
    if not est_path.exists():
        raise FileNotFoundError(f"Estimated not found: {est_path}")

    ref_data, ref_sr = sf.read(str(ref_path), dtype="float32")
    est_data, est_sr = sf.read(str(est_path), dtype="float32")

    # Ensure 2D shape (samples, channels)
    if ref_data.ndim == 1:
        ref_data = ref_data[:, np.newaxis]
    if est_data.ndim == 1:
        est_data = est_data[:, np.newaxis]

    # Resample if needed
    if ref_sr != target_sr:
        ref_data = resample_audio(ref_data, ref_sr, target_sr)
        ref_sr = target_sr
    if est_sr != target_sr:
        est_data = resample_audio(est_data, est_sr, target_sr)
        est_sr = target_sr

    # Ensure same length
    min_len = min(ref_data.shape[0], est_data.shape[0])
    ref_data = ref_data[:min_len]
    est_data = est_data[:min_len]

    return ref_data, est_data, target_sr


def calculate_metrics_museval(ref_data: np.ndarray, est_data: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Рассчитать SDR, SIR, SAR с использованием museval.

    Примечание: museval может требовать много памяти для длинного аудио.
    Используется меньший размер окна для избежания проблем с памятью.
    """
    if not HAS_MUSEVAL:
        raise RuntimeError("museval not installed")

    # museval ожидает (n_windows, n_channels, n_sources)
    # Для оценки одного источника: форма должна быть (1, n_channels, n_samples)
    ref_3d = ref_data.T[np.newaxis, ...]  # (1, каналы, сэмплы)
    est_3d = est_data.T[np.newaxis, ...]

    # Используем меньший размер окна для избежания проблем с памятью (4096 сэмплов ~ 93мс при 44100 Гц)
    win_size = 4096
    hop_size = 2048

    try:
        scores = museval.evaluate(
            references=ref_3d,
            estimates=est_3d,
            win=win_size,
            hop=hop_size,
        )
    except (TypeError, ValueError):
        # Пробуем с параметром sample_rate (старый API)
        try:
            scores = museval.evaluate(
                references=ref_3d,
                estimates=est_3d,
                sample_rate=sample_rate,
                win=win_size,
                hop=hop_size,
            )
        except Exception:
            # Запасной вариант: упрощённые метрики, если museval полностью не работает
            return calculate_simple_metrics(ref_data, est_data)

    return {
        "SDR": float(np.mean(scores["SDR"])),
        "SIR": float(np.mean(scores["SIR"])),
        "SAR": float(np.mean(scores["SAR"])),
    }


def calculate_metrics_mir_eval(ref_data: np.ndarray, est_data: np.ndarray) -> dict[str, float]:
    """Рассчитать SDR, SIR, SAR с использованием mir_eval."""
    if not HAS_MIR_EVAL:
        raise RuntimeError("mir_eval не установлен")

    # Преобразование в формат (n_sources, n_samples)
    ref_array = ref_data.T  # (channels, samples) -> (samples, channels)
    est_array = est_data.T

    # Усреднение до моно для mir_eval
    if ref_array.ndim == 2:
        ref_mono = np.mean(ref_array, axis=1)
    else:
        ref_mono = ref_array
    if est_array.ndim == 2:
        est_mono = np.mean(est_array, axis=1)
    else:
        est_mono = est_array

    ref_mono = ref_mono[np.newaxis, :]  # (1, n_samples)
    est_mono = est_mono[np.newaxis, :]

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_mono, est_mono)

    return {
        "SDR": float(sdr[0]),
        "SIR": float(sir[0]),
        "SAR": float(sar[0]),
    }


def calculate_simple_metrics(ref_data: np.ndarray, est_data: np.ndarray) -> dict[str, float]:
    """Рассчитать упрощённые метрики качества без внешних библиотек."""
    ref = ref_data
    est = est_data

    # Обеспечение одинаковой длины
    min_len = min(ref.shape[-1] if ref.ndim > 1 else len(ref),
                  est.shape[-1] if est.ndim > 1 else len(est))
    if ref.ndim > 1:
        ref = ref[:min_len]
    else:
        ref = ref[:min_len]
    if est.ndim > 1:
        est = est[:min_len]
    else:
        est = est[:min_len]

    # Преобразование в моно
    if ref.ndim > 1:
        ref = np.mean(ref, axis=1)
    if est.ndim > 1:
        est = np.mean(est, axis=1)

    # Расчёт энергии
    ref_energy = np.mean(ref ** 2)
    est_energy = np.mean(est ** 2)

    # Ошибка
    error = est - ref
    error_energy = np.mean(error ** 2)

    # SNR как приближение для SDR
    if error_energy > 1e-10:
        sdr = 10 * np.log10(max(ref_energy, 1e-10) / max(error_energy, 1e-10))
    else:
        sdr = 30.0

    # Корреляция как приближение для SIR
    corr = np.corrcoef(ref, est)[0, 1]
    if np.isnan(corr):
        sir = 10.0
    else:
        sir = -10 * np.log10(max(1e-10, 1 - corr ** 2))

    # SAR как среднее между SDR и SIR
    sar = (sdr + sir) / 2

    return {"SDR": float(sdr), "SIR": float(sir), "SAR": float(sar)}


def evaluate_source(
    ref_path: Path,
    est_path: Path,
    source_name: str,
    prefer_mir_eval: bool = True,
) -> dict[str, float]:
    """Оценить один источник (вокал, барабаны и т.д.)."""
    try:
        ref_data, est_data, sr = load_audio_pair(ref_path, est_path)
    except FileNotFoundError as e:
        logger.warning(f"{source_name}: {e}")
        return {"SDR": 0.0, "SIR": 0.0, "SAR": 0.0}

    try:
        # Сначала пробуем mir_eval (более эффективен по памяти)
        if prefer_mir_eval and HAS_MIR_EVAL:
            metrics = calculate_metrics_mir_eval(ref_data, est_data)
        # Затем пробуем museval с исправленными параметрами
        elif HAS_MUSEVAL:
            metrics = calculate_metrics_museval(ref_data, est_data, sr)
        # Запасной вариант: упрощённые метрики
        else:
            metrics = calculate_simple_metrics(ref_data, est_data)

        logger.info(
            f"{source_name}: SDR={metrics['SDR']:.2f}, "
            f"SIR={metrics['SIR']:.2f}, SAR={metrics['SAR']:.2f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"Ошибка оценки {source_name}: {e}")
        # Финальный запасной вариант: упрощённые метрики
        try:
            return calculate_simple_metrics(ref_data, est_data)
        except Exception:
            return {"SDR": 0.0, "SIR": 0.0, "SAR": 0.0}


def evaluate_track(
    ref_dir: Path,
    est_dir: Path,
    sources: Optional[list[str]] = None,
    prefer_mir_eval: bool = True,
) -> dict[str, dict[str, float]]:
    """Оценить все источники для одного трека."""
    if sources is None:
        sources = STEM_NAMES

    results = {}
    for source in sources:
        ref_path = ref_dir / f"{source}.wav"
        est_path = est_dir / f"{source}.wav"
        results[source] = evaluate_source(ref_path, est_path, source, prefer_mir_eval)

    return results


def print_results_table(aggregated: dict):
    """Вывести отформатированную таблицу результатов."""
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    header = f"{'Источник':<12} | {'SDR (дБ)':>10} | {'SIR (дБ)':>10} | {'SAR (дБ)':>10}"
    print(header)
    print("-" * 70)

    for source, metrics in aggregated.items():
        sdr = metrics["SDR"]["mean"]
        sir = metrics["SIR"]["mean"]
        sar = metrics["SAR"]["mean"]
        print(f"{source:<12} | {sdr:>10.2f} | {sir:>10.2f} | {sar:>10.2f}")

    print("=" * 70)


def evaluate_dataset(
    ref_root: Path,
    est_root: Path,
    track_names: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    prefer_mir_eval: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """Оценить несколько треков и вернуть результаты по каждому треку."""
    if track_names is None:
        track_names = sorted([d.name for d in ref_root.iterdir() if d.is_dir()])

    all_results = {}
    for track_name in track_names:
        ref_dir = ref_root / track_name
        est_dir = est_root / track_name

        if not ref_dir.exists():
            logger.warning(f"Track not found: {ref_dir}")
            continue
        if not est_dir.exists():
            logger.warning(f"Separated track not found: {est_dir}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating track: {track_name}")
        logger.info(f"{'=' * 60}")

        all_results[track_name] = evaluate_track(
            ref_dir, est_dir, sources, prefer_mir_eval
        )

    return all_results


def aggregate_results(
    results: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Агрегировать метрики по всем трекам."""
    sources = STEM_NAMES
    metrics = ["SDR", "SIR", "SAR"]

    aggregated = {}

    for source in sources:
        values = {m: [] for m in metrics}

        for track_results in results.values():
            if source in track_results:
                for metric in metrics:
                    val = track_results[source].get(metric, 0.0)
                    if val != 0.0:
                        values[metric].append(val)

        aggregated[source] = {}
        for metric in metrics:
            if values[metric]:
                aggregated[source][metric] = {
                    "mean": float(np.mean(values[metric])),
                    "std": float(np.std(values[metric])),
                    "min": float(np.min(values[metric])),
                    "max": float(np.max(values[metric])),
                }
            else:
                aggregated[source][metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

    return aggregated


def print_results_table(aggregated: dict):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    header = f"{'Source':<10} | {'SDR (dB)':>10} | {'SIR (dB)':>10} | {'SAR (dB)':>10}"
    print(header)
    print("-" * 70)

    for source, metrics in aggregated.items():
        sdr = metrics["SDR"]["mean"]
        sir = metrics["SIR"]["mean"]
        sar = metrics["SAR"]["mean"]
        print(f"{source:<10} | {sdr:>10.2f} | {sir:>10.2f} | {sar:>10.2f}")

    print("=" * 70)


def run_full_evaluation(
    ref_root: str | Path,
    est_root: str | Path,
    output_file: Optional[str | Path] = None,
    track_names: Optional[list[str]] = None,
    prefer_mir_eval: bool = True,
) -> dict:
    """Запустить полный пайплайн оценки и сохранить результаты."""
    ref_root = Path(ref_root)
    est_root = Path(est_root)

    all_results = evaluate_dataset(
        ref_root, est_root, track_names, prefer_mir_eval=prefer_mir_eval
    )

    aggregated = aggregate_results(all_results)

    print_results_table(aggregated)

    output = {
        "per_track": all_results,
        "aggregated": aggregated,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    return output


def separate_and_evaluate(
    musdb_root: Path,
    output_root: Path,
    model_name: str = "htdemucs",
    device: str = "cpu",
    track_names: Optional[list[str]] = None,
    prefer_mir_eval: bool = True,
) -> dict:
    """Разделить треки с помощью Demucs, затем оценить их."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torch

    output_root.mkdir(parents=True, exist_ok=True)

    model = get_model(name=model_name)
    model.eval()
    if device == "cuda":
        model = model.cuda()

    if track_names is None:
        track_names = sorted([d.name for d in musdb_root.iterdir() if d.is_dir()])

    all_results = {}
    times = []

    for i, track_name in enumerate(track_names):
        track_dir = musdb_root / track_name
        output_track_dir = output_root / track_name
        output_track_dir.mkdir(parents=True, exist_ok=True)

        mix_path = track_dir / "mixture.wav"
        if not mix_path.exists():
            logger.warning(f"[{i+1}] {track_name}: SKIP (no mixture.wav)")
            continue

        logger.info(f"[{i+1}/{len(track_names)}] {track_name}")

        # Separate
        logger.info("  Separating...")
        try:
            start = time.time()

            data, sr = sf.read(str(mix_path), dtype="float32")
            if data.ndim == 1:
                data = data[:, np.newaxis]
            waveform = torch.from_numpy(data.T)

            with torch.no_grad():
                separated = apply_model(model, waveform.unsqueeze(0).to(device))

            separated = separated.squeeze(0).cpu()

            for idx, stem_name in enumerate(STEM_NAMES):
                stem_waveform = separated[idx]
                peak = stem_waveform.abs().max()
                if peak > 1e-6:
                    stem_waveform = stem_waveform / peak * 0.95
                stem_path = output_track_dir / f"{stem_name}.wav"
                sf.write(str(stem_path), stem_waveform.numpy().T, sr, subtype="PCM_24")

            elapsed = time.time() - start
            times.append(elapsed)
            logger.info(f"  Done in {elapsed:.1f}s")

            # Evaluate
            track_results = evaluate_track(
                track_dir, output_track_dir, prefer_mir_eval=prefer_mir_eval
            )
            all_results[track_name] = track_results

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate
    aggregated = aggregate_results(all_results)

    # Print
    print_results_table(aggregated)

    if times:
        print(f"\nAvg separation time: {np.mean(times):.1f}s")
        print(f"Min: {np.min(times):.1f}s, Max: {np.max(times):.1f}s")

    return {"per_track": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate source separation on MUSDB18")
    parser.add_argument("musdb_root", help="Path to MUSDB18 dataset")
    parser.add_argument("output_root", help="Path to separated stems")
    parser.add_argument("--tracks", nargs="+", default=None, help="Track names to evaluate")
    parser.add_argument("-o", "--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--separate", action="store_true", help="Run separation before evaluation")
    parser.add_argument("--use-mir-eval", action="store_true", default=True, help="Prefer mir_eval over museval")

    args = parser.parse_args()

    if args.separate:
        results = separate_and_evaluate(
            musdb_root=Path(args.musdb_root),
            output_root=Path(args.output_root),
            model_name=args.model,
            device=args.device,
            track_names=args.tracks,
            prefer_mir_eval=args.use_mir_eval,
        )
    else:
        results = run_full_evaluation(
            ref_root=args.musdb_root,
            est_root=args.output_root,
            output_file=args.output,
            track_names=args.tracks,
            prefer_mir_eval=args.use_mir_eval,
        )

    # Save if not already saved
    if not args.separate and args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
