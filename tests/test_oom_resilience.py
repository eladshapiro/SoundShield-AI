"""A full GPU must shrink batches, not silently drop the neural models (cuda_compat.halve_on_oom)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuda_compat import halve_on_oom, is_cuda_oom  # noqa: E402
from emotion_models import EmotionAnalyzer  # noqa: E402

OOM = 'CUDA out of memory. Tried to allocate 500.00 MiB.'


def small_batches_only(limit):
    calls = []

    def fn(items):
        calls.append(len(items))
        if len(items) > limit:
            raise RuntimeError(OOM)
        return np.asarray(items, dtype=np.float32) * 10
    return fn, calls


class TestHalveOnOOM(unittest.TestCase):

    def test_splits_until_it_fits_and_keeps_order(self):
        fn, calls = small_batches_only(2)
        out = halve_on_oom(fn, list(range(7)))
        np.testing.assert_array_equal(out, np.arange(7) * 10)
        self.assertLessEqual(max(c for c in calls if c <= 2), 2)

    def test_joins_tuples_and_lists(self):
        def fn(items):
            if len(items) > 1:
                raise RuntimeError(OOM)
            return [{'x': items[0]}], np.array([[items[0]]])
        probs, hidden = halve_on_oom(fn, [1, 2, 3])
        self.assertEqual([p['x'] for p in probs], [1, 2, 3])
        np.testing.assert_array_equal(hidden[:, 0], [1, 2, 3])

    def test_other_errors_and_single_items_still_raise(self):
        def bad(items):
            raise RuntimeError('shape mismatch')
        with self.assertRaises(RuntimeError):
            halve_on_oom(bad, [1, 2, 3, 4])
        fn, _ = small_batches_only(0)
        with self.assertRaises(RuntimeError):
            halve_on_oom(fn, [1, 2])

    def test_is_cuda_oom(self):
        self.assertTrue(is_cuda_oom(RuntimeError(OOM)))
        self.assertTrue(is_cuda_oom(RuntimeError('CUDA error: out of memory')))
        self.assertFalse(is_cuda_oom(ValueError(OOM)))
        self.assertFalse(is_cuda_oom(RuntimeError('device-side assert')))


class FakeDim:
    def load(self):
        return True

    def predict(self, chunks, return_hidden=False):
        if len(chunks) > 2:
            raise RuntimeError(OOM)
        dims = np.array([[float(c.mean()), 0.5, 0.5] for c in chunks], dtype=np.float32)
        if return_hidden:
            return dims, np.repeat(dims[:, :1], 4, axis=1)
        return dims


class FakeCat:
    def load(self):
        return True

    def predict(self, chunks, return_hidden=False):
        if len(chunks) > 2:
            raise RuntimeError(OOM)
        probs = [{'ang': float(c.mean()), 'hap': 0.1, 'neu': 0.1, 'sad': 0.1} for c in chunks]
        if return_hidden:
            return probs, np.full((len(chunks), 3), 0.5, dtype=np.float32)
        return probs


class TestEmotionTimelineUnderMemoryPressure(unittest.TestCase):

    def test_timeline_survives_oom_with_identical_values(self):
        ea = EmotionAnalyzer(use_dimensional=False, use_categorical=False)
        ea.dim, ea.cat = FakeDim(), FakeCat()
        levels = np.linspace(0.0, 0.9, 10, dtype=np.float32)
        audio = np.concatenate([np.full(int(2.5 * 16000), v, dtype=np.float32) for v in levels])
        et = ea.timeline(audio, 16000, batch_size=16, embeddings=True)
        self.assertIsNotNone(et)
        n = len(et.starts)
        self.assertEqual(len(et.arousal), n)
        self.assertEqual(et.embeddings['w2v2'].shape, (n, 4))
        self.assertEqual(et.embeddings['hubert'].shape, (n, 3))
        # window i spans pieces i and i+1 -> its mean is the average of the two levels
        expected = (levels[:-1] + levels[1:]) / 2
        np.testing.assert_allclose(et.arousal, expected[:n], atol=1e-5)
        np.testing.assert_allclose(et.categorical['ang'], expected[:n], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
