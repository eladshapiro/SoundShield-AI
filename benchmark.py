#!/usr/bin/env python3
"""
Performance Benchmark for SoundShield-AI

Measures inference latency and throughput for each detector module.
Usage: python benchmark.py [--duration 5] [--iterations 10]
"""

import argparse
import time
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from config import config


def generate_test_audio(sr: int = 22050, duration: float = 5.0) -> np.ndarray:
    """Generate realistic test audio with mixed signals."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix: speech-like tone + some noise + cry-like bursts
    signal = (
        0.1 * np.sin(2 * np.pi * 150 * t) +
        0.05 * np.sin(2 * np.pi * 300 * t) +
        0.02 * np.random.randn(len(t))
    )
    # Add a "cry burst" in the middle
    burst_start = int(len(t) * 0.4)
    burst_end = int(len(t) * 0.6)
    signal[burst_start:burst_end] += 0.3 * np.sin(2 * np.pi * 450 * t[burst_start:burst_end])
    return signal.astype(np.float32)


def benchmark_detector(name: str, func, audio: np.ndarray, sr: int,
                       iterations: int) -> dict:
    """Benchmark a single detector."""
    # Warmup
    try:
        func(audio, sr)
    except Exception as e:
        return {'name': name, 'error': str(e)}

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(audio, sr)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times_np = np.array(times)
    audio_duration = len(audio) / sr

    return {
        'name': name,
        'iterations': iterations,
        'audio_duration_s': round(audio_duration, 2),
        'avg_ms': round(float(np.mean(times_np)), 2),
        'p50_ms': round(float(np.percentile(times_np, 50)), 2),
        'p95_ms': round(float(np.percentile(times_np, 95)), 2),
        'min_ms': round(float(np.min(times_np)), 2),
        'max_ms': round(float(np.max(times_np)), 2),
        'realtime_factor': round(audio_duration / (float(np.mean(times_np)) / 1000), 1),
    }


def run_benchmarks(duration: float = 5.0, iterations: int = 10):
    """Run benchmarks for all detector modules."""
    sr = config.audio.sample_rate
    audio = generate_test_audio(sr, duration)

    print(f"\nSoundShield-AI Performance Benchmark")
    print(f"{'=' * 60}")
    print(f"Audio: {duration}s @ {sr}Hz | Iterations: {iterations}")
    print(f"{'=' * 60}\n")

    results = []

    # AudioAnalyzer
    try:
        from audio_analyzer import AudioAnalyzer
        aa = AudioAnalyzer()
        r = benchmark_detector('AudioAnalyzer.extract_features',
                               aa.extract_features, audio, sr, iterations)
        results.append(r)
    except Exception as e:
        results.append({'name': 'AudioAnalyzer', 'error': str(e)})

    # CryDetector
    try:
        from cry_detector import CryDetector
        cd = CryDetector()
        r = benchmark_detector('CryDetector.detect_cry_segments',
                               cd.detect_cry_segments, audio, sr, iterations)
        results.append(r)
    except Exception as e:
        results.append({'name': 'CryDetector', 'error': str(e)})

    # ViolenceDetector
    try:
        from violence_detector import ViolenceDetector
        vd = ViolenceDetector()
        r = benchmark_detector('ViolenceDetector.detect_violence_segments',
                               vd.detect_violence_segments, audio, sr, iterations)
        results.append(r)
    except Exception as e:
        results.append({'name': 'ViolenceDetector', 'error': str(e)})

    # EmotionDetector
    try:
        from emotion_detector import EmotionDetector
        ed = EmotionDetector()

        def emotion_pipeline(audio_data, sample_rate):
            features = ed.calculate_emotion_features(audio_data, sample_rate)
            return ed.detect_emotion(features)

        r = benchmark_detector('EmotionDetector (full pipeline)',
                               emotion_pipeline, audio, sr, iterations)
        results.append(r)
    except Exception as e:
        results.append({'name': 'EmotionDetector', 'error': str(e)})

    # NeglectDetector
    try:
        from neglect_detector import NeglectDetector
        nd = NeglectDetector()

        def neglect_pipeline(audio_data, sample_rate):
            return nd.detect_neglect_patterns(audio_data, sample_rate)

        r = benchmark_detector('NeglectDetector.detect_neglect_patterns',
                               neglect_pipeline, audio, sr, iterations)
        results.append(r)
    except Exception as e:
        results.append({'name': 'NeglectDetector', 'error': str(e)})

    # ONNX (if available)
    try:
        from model_optimizer import optimizer
        if optimizer.load_onnx_model():
            audio_16k = np.random.randn(16000).astype(np.float32)
            onnx_result = optimizer.benchmark(audio_16k, sr=16000, iterations=iterations)
            results.append({
                'name': 'HuBERT ONNX Inference',
                **onnx_result,
            })
    except Exception:
        pass

    # Print results
    print(f"{'Detector':<45} {'Avg (ms)':>10} {'P95 (ms)':>10} {'RT Factor':>10}")
    print(f"{'-' * 75}")

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<45} {'ERROR':>10}   {r['error'][:25]}")
        else:
            print(f"{r['name']:<45} {r['avg_ms']:>10.1f} {r['p95_ms']:>10.1f} {r['realtime_factor']:>9.1f}x")

    print(f"\n{'=' * 60}")
    print("RT Factor > 1.0x means faster than real-time (good)")
    print("RT Factor < 1.0x means slower than real-time (needs optimization)")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SoundShield-AI Benchmark')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Test audio duration in seconds')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per benchmark')
    args = parser.parse_args()
    run_benchmarks(args.duration, args.iterations)
