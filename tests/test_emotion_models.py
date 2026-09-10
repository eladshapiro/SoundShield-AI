"""Unit tests for emotion_models (timeline container + decision rules)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_models import EmotionAnalyzer, EmotionTimeline, aggressive_rule, anger_rule  # noqa: E402
from config import config  # noqa: E402


def make_emotion_timeline(arousal, dominance, valence, p_ang=None, window=5.0, hop=2.5):
    n = len(arousal)
    starts = np.arange(n) * hop
    cat = {'ang': np.asarray(p_ang, dtype=np.float32)} if p_ang is not None else {}
    return EmotionTimeline(starts=starts, window=window, duration=starts[-1] + window,
                           arousal=np.asarray(arousal, dtype=np.float32),
                           dominance=np.asarray(dominance, dtype=np.float32),
                           valence=np.asarray(valence, dtype=np.float32), categorical=cat)


class TestEmotionTimeline(unittest.TestCase):

    def test_flags_and_scores(self):
        tl = make_emotion_timeline([0.2, 0.9], [0.3, 0.8], [0.6, 0.2], p_ang=[0.1, 0.9])
        self.assertTrue(tl.has_dimensional)
        self.assertTrue(tl.has_categorical)
        self.assertAlmostEqual(tl.score('arousal', 0, 2.5), 0.2, places=5)
        self.assertAlmostEqual(tl.score('arousal'), 0.9, places=5)
        self.assertAlmostEqual(tl.score('ang'), 0.9, places=5)
        d = tl.to_dict()
        self.assertEqual(len(d['arousal']), 2)
        self.assertIn('p_ang', d)

    def test_without_dimensional_model(self):
        tl = EmotionTimeline(starts=np.array([0.0]), window=5.0, duration=5.0,
                             arousal=np.zeros(0), dominance=np.zeros(0), valence=np.zeros(0),
                             categorical={'ang': np.array([0.8])})
        self.assertFalse(tl.has_dimensional)
        self.assertEqual(tl.score('arousal'), 0.0)
        self.assertAlmostEqual(tl.score('ang'), 0.8, places=5)


class TestRules(unittest.TestCase):

    def setUp(self):
        self.cfg = config.advanced
        self.vcfg = config.violence

    def test_anger_rule_needs_dims_plus_valence_or_hubert(self):
        a, d = self.cfg.arousal_threshold, self.cfg.dominance_threshold
        # excited but high-valence speech (happy) and HuBERT silent -> not anger
        flag, _, _ = anger_rule(0.1, a + 0.1, d + 0.1, 0.9)
        self.assertFalse(flag)
        # low valence -> aggression
        flag, conf, emotion = anger_rule(0.1, a + 0.1, d + 0.1, self.cfg.valence_max)
        self.assertTrue(flag)
        self.assertEqual(emotion, 'aggression')
        self.assertGreater(conf, 0)
        # HuBERT agrees -> anger
        flag, conf, emotion = anger_rule(self.cfg.anger_prob_threshold, a + 0.1, d + 0.1, 0.9)
        self.assertTrue(flag)
        self.assertEqual(emotion, 'anger')
        # calm dims -> never
        flag, _, _ = anger_rule(0.99, 0.0, 0.0, 0.0)
        self.assertFalse(flag)

    def test_anger_rule_without_dimensional_model_uses_hubert(self):
        self.assertTrue(anger_rule(self.cfg.anger_prob_threshold, None, None, None)[0])
        self.assertFalse(anger_rule(self.cfg.anger_prob_threshold - 0.01, None, None, None)[0])

    def test_aggressive_rule(self):
        a, d, v = self.vcfg.aggressive_arousal, self.vcfg.aggressive_dominance, self.vcfg.aggressive_valence_max
        self.assertTrue(aggressive_rule(a, d, v)[0])
        self.assertFalse(aggressive_rule(a, d, v + 0.01)[0])
        self.assertFalse(aggressive_rule(a - 0.01, d, v)[0])
        self.assertTrue(aggressive_rule(a, d, None)[0])


class TestEmotionAnalyzerWindows(unittest.TestCase):

    def test_windows(self):
        an = EmotionAnalyzer(window=5.0, hop=2.5, use_dimensional=False, use_categorical=False)
        starts = an._windows(16000 * 12)
        self.assertEqual(starts[0], 0.0)
        self.assertAlmostEqual(starts[-1], 7.0, places=3)
        self.assertEqual(list(an._windows(16000 * 2)), [0.0])

    def test_no_models_returns_none(self):
        an = EmotionAnalyzer(use_dimensional=False, use_categorical=False)
        self.assertFalse(an.available)
        self.assertIsNone(an.timeline(np.zeros(16000 * 6, dtype=np.float32), 16000))


if __name__ == '__main__':
    unittest.main()
