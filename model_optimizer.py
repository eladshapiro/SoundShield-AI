"""
Model Optimizer for SoundShield-AI

Provides ONNX export and optimized inference for HuBERT emotion model.
Supports: PyTorch -> ONNX export, ONNX Runtime inference, quantization.
"""

import logging
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

MODELS_DIR = Path('models')
ONNX_FILENAME = 'hubert_emotion.onnx'
ONNX_QUANTIZED_FILENAME = 'hubert_emotion_int8.onnx'


class ModelOptimizer:
    """Export and optimize ML models for production inference."""

    def __init__(self):
        self.onnx_model_path: Optional[str] = None
        self.onnx_session = None
        self._onnx_available = False

        try:
            import onnxruntime
            self._onnx_available = True
        except ImportError:
            logger.info("ONNX Runtime not installed — ONNX inference disabled")

    def export_hubert_to_onnx(self, output_dir: str = None,
                               quantize: bool = True) -> Optional[str]:
        """Export HuBERT emotion model from PyTorch to ONNX format.

        Args:
            output_dir: Directory to save ONNX model (default: models/)
            quantize: Apply INT8 dynamic quantization after export

        Returns:
            Path to exported ONNX model, or None on failure.
        """
        output_dir = Path(output_dir) if output_dir else MODELS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

            model_name = config.advanced.hubert_model
            logger.info(f"Exporting {model_name} to ONNX...")

            # Load PyTorch model
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            model.eval()

            # Create dummy input (1 second of audio at 16kHz)
            dummy_input = torch.randn(1, 16000)

            # Export to ONNX
            onnx_path = str(output_dir / ONNX_FILENAME)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_values'],
                output_names=['logits'],
                dynamic_axes={
                    'input_values': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'},
                },
            )
            logger.info(f"ONNX model exported to {onnx_path}")

            # Verify
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")

            # Quantize if requested
            if quantize:
                quantized_path = self._quantize_onnx(onnx_path, output_dir)
                if quantized_path:
                    return quantized_path

            return onnx_path

        except ImportError as e:
            logger.error(f"Missing dependency for ONNX export: {e}")
            return None
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None

    def _quantize_onnx(self, model_path: str, output_dir: Path) -> Optional[str]:
        """Apply dynamic INT8 quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = str(output_dir / ONNX_QUANTIZED_FILENAME)
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )

            original_size = os.path.getsize(model_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100

            logger.info(f"Quantized: {original_size:.1f}MB -> {quantized_size:.1f}MB "
                        f"({reduction:.1f}% reduction)")
            return quantized_path

        except ImportError:
            logger.warning("onnxruntime.quantization not available — skipping quantization")
            return None
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return None

    def load_onnx_model(self, model_path: str = None) -> bool:
        """Load an ONNX model for inference.

        Auto-discovers models in models/ directory if no path given.
        Priority: quantized > full-precision.
        """
        if not self._onnx_available:
            logger.warning("ONNX Runtime not installed")
            return False

        import onnxruntime as ort

        if model_path is None:
            # Auto-discover
            quantized = MODELS_DIR / ONNX_QUANTIZED_FILENAME
            full = MODELS_DIR / ONNX_FILENAME
            if quantized.exists():
                model_path = str(quantized)
            elif full.exists():
                model_path = str(full)
            else:
                logger.warning("No ONNX model found in models/")
                return False

        try:
            # Configure session for optimal performance
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = os.cpu_count() or 4

            providers = ['CPUExecutionProvider']
            # Try GPU if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

            self.onnx_session = ort.InferenceSession(
                model_path, sess_options=opts, providers=providers
            )
            self.onnx_model_path = model_path
            provider = self.onnx_session.get_providers()[0]
            logger.info(f"ONNX model loaded from {model_path} (provider: {provider})")
            return True

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False

    def predict_emotion_onnx(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """Run emotion prediction using ONNX model.

        Args:
            audio: Audio waveform (numpy array, mono, 16kHz)
            sr: Sample rate

        Returns:
            Dict with emotion label, confidence, and all scores.
        """
        if self.onnx_session is None:
            return {'error': 'ONNX model not loaded'}

        try:
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Add batch dimension if needed
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            # Run inference
            start_time = time.perf_counter()
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            logits = self.onnx_session.run([output_name], {input_name: audio})[0]
            inference_ms = (time.perf_counter() - start_time) * 1000

            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            probs = probs[0]

            # Map to labels (HuBERT superb-er has 4 classes)
            labels = ['neu', 'hap', 'ang', 'sad']
            scores = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
            predicted_idx = int(np.argmax(probs))
            predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else 'unknown'

            return {
                'label': predicted_label,
                'confidence': float(probs[predicted_idx]),
                'scores': scores,
                'inference_ms': round(inference_ms, 2),
                'backend': 'onnx',
            }

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return {'error': str(e)}

    def benchmark(self, audio: np.ndarray, sr: int = 16000,
                  iterations: int = 10) -> Dict:
        """Benchmark ONNX inference speed.

        Returns average latency, throughput, and percentiles.
        """
        if self.onnx_session is None:
            return {'error': 'ONNX model not loaded'}

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        # Warmup
        for _ in range(3):
            self.onnx_session.run([output_name], {input_name: audio})

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.onnx_session.run([output_name], {input_name: audio})
            times.append((time.perf_counter() - start) * 1000)

        times_np = np.array(times)
        audio_duration = audio.shape[1] / sr

        return {
            'iterations': iterations,
            'audio_duration_s': round(audio_duration, 2),
            'avg_ms': round(float(np.mean(times_np)), 2),
            'p50_ms': round(float(np.percentile(times_np, 50)), 2),
            'p95_ms': round(float(np.percentile(times_np, 95)), 2),
            'p99_ms': round(float(np.percentile(times_np, 99)), 2),
            'min_ms': round(float(np.min(times_np)), 2),
            'max_ms': round(float(np.max(times_np)), 2),
            'realtime_factor': round(audio_duration / (float(np.mean(times_np)) / 1000), 1),
            'model_path': self.onnx_model_path,
        }


# Global instance
optimizer = ModelOptimizer()
