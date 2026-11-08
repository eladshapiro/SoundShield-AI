"""Learned decision heads (event_heads.py): features, JSON round trip, detector wiring.

No model is loaded: timelines and embeddings are built by hand.
"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import event_heads  # noqa: E402
from event_heads import LinearHead, window_features  # noqa: E402
from audio_tagger import TagTimeline  # noqa: E402
from emotion_models import EmotionTimeline  # noqa: E402
from advanced_analyzer import AdvancedAnalyzer  # noqa: E402
from config import config  # noqa: E402

LABELS = ['Speech', 'Baby cry, infant cry', 'Screaming', 'Laughter', 'Silence']


def tag_timeline(duration, speech=0.9, emb_dim=8, with_emb=True):
    starts = np.arange(0, max(1, int(duration) - 2), 1.0)
    probs = np.zeros((len(starts), len(LABELS)), dtype=np.float32)
    probs[:, 0] = speech
    emb = (np.arange(len(starts) * emb_dim, dtype=np.float32).reshape(len(starts), emb_dim) / 100.0
           ).astype(np.float16) if with_emb else None
    return TagTimeline(starts=starts, window=3.0, probs=probs, labels=LABELS, duration=duration, embeddings=emb)


def emotion_timeline(duration, w2v2_dim=4, with_emb=True):
    starts = np.arange(0, max(1, int(duration) - 4), 2.5)
    n = len(starts)
    emb = {'w2v2': np.tile(np.arange(n, dtype=np.float32)[:, None], (1, w2v2_dim))} if with_emb else {}
    return EmotionTimeline(starts=starts, window=5.0, duration=duration,
                           arousal=np.full(n, 0.3, np.float32), dominance=np.full(n, 0.4, np.float32),
                           valence=np.full(n, 0.5, np.float32),
                           categorical={k: np.full(n, 0.25, np.float32) for k in ('ang', 'hap', 'neu', 'sad')},
                           embeddings=emb)


def head_on_w2v2(threshold=0.5, dim=4):
    """P(anger) = sigmoid(sum(w2v2) - 2*dim): above 0.5 once the window index exceeds 2."""
    return LinearHead(name='anger', blocks=('w2v2',), mean=np.zeros(dim, np.float32),
                      scale=np.ones(dim, np.float32), coef=np.ones(dim, np.float32), bias=-2.0 * dim,
                      threshold=threshold, meta={})


class HeadTestCase(unittest.TestCase):
    """Isolate the module-level head cache and the config knobs each test touches."""

    def setUp(self):
        self._saved = (dict(event_heads._HEADS), config.advanced.use_learned_heads,
                       config.advanced.anger_head_threshold, config.advanced.heads_dir,
                       config.violence.scream_head_threshold)
        event_heads._HEADS.clear()
        event_heads._HEADS['scream'] = None        # tests opt in to a scream head explicitly

    def tearDown(self):
        heads, use, thr, hdir, sthr = self._saved
        event_heads._HEADS.clear()
        event_heads._HEADS.update(heads)
        config.advanced.use_learned_heads = use
        config.advanced.anger_head_threshold = thr
        config.advanced.heads_dir = hdir
        config.violence.scream_head_threshold = sthr

    def install(self, head, name='anger'):
        event_heads._HEADS[name] = head


class TestWindowFeatures(HeadTestCase):

    def test_blocks_and_shapes(self):
        et, tl = emotion_timeline(30), tag_timeline(30)
        X = window_features(et, tl, ('w2v2', 'ast', 'dims', 'cat', 'tags'))
        self.assertEqual(X.shape, (len(et.starts), 4 + 8 + 3 + 4 + len(event_heads.TAG_GROUPS)))
        # ast block = mean of the tagger embeddings overlapping the 5 s window
        lo, hi = event_heads._tag_ranges(tl, np.asarray(et.starts), et.window)[0]
        np.testing.assert_allclose(X[0, 4:12], tl.embeddings[lo:hi].astype(np.float32).mean(axis=0), rtol=1e-3)
        # tags block: speech group max
        self.assertAlmostEqual(float(X[0, 19 + event_heads.TAG_GROUPS.index('speech')]), 0.9, places=5)

    def test_missing_block_gives_none(self):
        self.assertIsNone(window_features(emotion_timeline(30, with_emb=False), tag_timeline(30), ('w2v2',)))
        self.assertIsNone(window_features(emotion_timeline(30), tag_timeline(30, with_emb=False), ('ast',)))
        self.assertIsNone(window_features(emotion_timeline(30), None, ('tags',)))


class TestLinearHead(HeadTestCase):

    def test_json_round_trip(self):
        head = head_on_w2v2(threshold=0.7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'anger.json'
            head.to_json(path)
            back = LinearHead.from_json(path)
        self.assertEqual(back.blocks, ('w2v2',))
        self.assertAlmostEqual(back.threshold, 0.7)
        X = np.random.RandomState(0).randn(5, 4).astype(np.float32)
        np.testing.assert_allclose(back.predict(X), head.predict(X), rtol=1e-5)

    def test_get_head_reads_heads_dir_and_respects_switch(self):
        with tempfile.TemporaryDirectory() as d:
            head_on_w2v2().to_json(Path(d) / 'anger.json')
            config.advanced.heads_dir = d
            self.assertIsNotNone(event_heads.get_head('anger'))
            event_heads._HEADS.clear()
            config.advanced.use_learned_heads = False
            self.assertIsNone(event_heads.get_head('anger'))
            self.assertFalse(event_heads.wants_embeddings())

    def test_threshold_override(self):
        self.install(head_on_w2v2(threshold=0.6))
        config.advanced.anger_head_threshold = float('nan')
        self.assertAlmostEqual(event_heads.threshold('anger'), 0.6)
        config.advanced.anger_head_threshold = 0.9
        self.assertAlmostEqual(event_heads.threshold('anger'), 0.9)


class TestDetectorWiring(HeadTestCase):

    def setUp(self):
        super().setUp()
        config.advanced.use_learned_heads = True
        config.advanced.anger_head_threshold = float('nan')
        self.an = AdvancedAnalyzer(use_whisper=False, use_transformer_emotion=False)

    def test_head_decides_when_installed(self):
        self.install(head_on_w2v2())
        et, tl = emotion_timeline(40), tag_timeline(40)
        res = self.an.concerning_emotions_from_timeline(et, tl)
        p = et.heads['anger']
        self.assertTrue(len(res) >= 1)
        first_hot = float(et.starts[int(np.argmax(p >= 0.5))])
        self.assertAlmostEqual(res[0]['start_time'], first_hot)
        self.assertEqual(res[0]['detected_emotion'], 'anger')
        self.assertIn('anger_head', res[0]['all_scores'])
        self.assertIn('head_anger', et.to_dict())

    def test_speech_gate_still_applies(self):
        self.install(head_on_w2v2())
        self.assertEqual(self.an.concerning_emotions_from_timeline(emotion_timeline(40), tag_timeline(40, speech=0.1)), [])

    def test_falls_back_to_rules_without_embeddings(self):
        self.install(head_on_w2v2())
        et = emotion_timeline(40, with_emb=False)     # calm dims, HuBERT 0.25: the rules flag nothing
        self.assertEqual(self.an.concerning_emotions_from_timeline(et, tag_timeline(40)), [])
        self.assertEqual(et.heads, {})

    def test_mismatched_head_falls_back_instead_of_failing(self):
        self.install(head_on_w2v2(dim=7))                # trained on another model: 7 dims, timeline has 4
        et = emotion_timeline(40)
        self.assertEqual(self.an.concerning_emotions_from_timeline(et, tag_timeline(40)), [])
        self.assertEqual(et.heads, {})

    def test_threshold_from_config(self):
        self.install(head_on_w2v2())
        config.advanced.anger_head_threshold = 1.5          # unreachable: nothing may fire
        self.assertEqual(self.an.concerning_emotions_from_timeline(emotion_timeline(40), tag_timeline(40)), [])


class TestViolenceWiring(HeadTestCase):

    def setUp(self):
        super().setUp()
        config.advanced.use_learned_heads = True
        config.advanced.anger_head_threshold = float('nan')
        from violence_detector import ViolenceDetector
        self.vd = ViolenceDetector()
        self.sr = 22050
        self.audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(40 * self.sr) / self.sr)).astype(np.float32)

    def test_aggressive_tone_needs_head_and_arousal(self):
        self.install(head_on_w2v2())
        et = emotion_timeline(40)
        et.arousal[:] = config.violence.aggressive_arousal + 0.1
        segs = self.vd._detect_violence_segments_ml(self.audio, self.sr, tag_timeline(40), et)
        self.assertTrue(any('aggressive_tone' in s['violence_types'] for s in segs))
        et_calm = emotion_timeline(40)
        et_calm.arousal[:] = config.violence.aggressive_arousal - 0.1     # angry words, calm voice: not violence
        segs = self.vd._detect_violence_segments_ml(self.audio, self.sr, tag_timeline(40), et_calm)
        self.assertFalse(any('aggressive_tone' in s['violence_types'] for s in segs))

    def screaming_tags(self, duration=40):
        tl = tag_timeline(duration, speech=0.0)
        tl.probs[:, LABELS.index('Screaming')] = 0.9          # the raw tag says "scream" everywhere
        return tl

    def test_scream_head_replaces_the_raw_scream_tag(self):
        quiet = LinearHead(name='scream', blocks=('w2v2',), mean=np.zeros(4, np.float32),
                           scale=np.ones(4, np.float32), coef=np.zeros(4, np.float32), bias=-5.0,
                           threshold=0.5, meta={})                       # P ~ 0.007 everywhere
        self.install(quiet, 'scream')
        segs = self.vd._detect_violence_segments_ml(self.audio, self.sr, self.screaming_tags(), emotion_timeline(40))
        self.assertFalse(any('shouting' in s['violence_types'] for s in segs))
        self.install(LinearHead(name='scream', blocks=('w2v2',), mean=np.zeros(4, np.float32),
                                scale=np.ones(4, np.float32), coef=np.ones(4, np.float32), bias=-8.0,
                                threshold=0.5, meta={}), 'scream')        # fires from window 3 on
        segs = self.vd._detect_violence_segments_ml(self.audio, self.sr, self.screaming_tags(), emotion_timeline(40))
        shout = [s for s in segs if 'shouting' in s['violence_types']]
        self.assertTrue(shout)
        self.assertGreaterEqual(shout[0]['start_time'], 5.0)

    def test_raw_scream_tag_without_head(self):
        segs = self.vd._detect_violence_segments_ml(self.audio, self.sr, self.screaming_tags(), emotion_timeline(40))
        self.assertTrue(any('shouting' in s['violence_types'] for s in segs))


class TestShippedHead(HeadTestCase):

    def test_shipped_head_file_is_valid(self):
        path = event_heads.HEADS_DIR / 'anger.json'
        if not path.exists():
            self.skipTest('no shipped head')
        head = LinearHead.from_json(path)
        self.assertEqual(len(head.mean), len(head.coef))
        self.assertEqual(len(head.scale), len(head.coef))
        self.assertTrue(0.0 < head.threshold < 1.0)
        self.assertTrue(all(b in ('w2v2', 'hubert', 'ast', 'dims', 'cat', 'tags') for b in head.blocks))
        self.assertFalse(any(math.isnan(float(x)) for x in head.coef))


if __name__ == '__main__':
    unittest.main()
