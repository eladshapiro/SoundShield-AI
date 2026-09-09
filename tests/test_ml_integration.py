"""Detectors driven by (synthetic) model timelines — the ML-first code paths.

No model is loaded: TagTimeline / EmotionTimeline objects are built by hand so
the tests are fast and deterministic.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_tagger import TagTimeline  # noqa: E402
from emotion_models import EmotionTimeline  # noqa: E402
from cry_detector import CryDetector  # noqa: E402
from violence_detector import ViolenceDetector  # noqa: E402
from neglect_detector import NeglectDetector  # noqa: E402
from advanced_analyzer import AdvancedAnalyzer  # noqa: E402
from inappropriate_language_detector import InappropriateLanguageDetector  # noqa: E402
from config import config  # noqa: E402

SR = 22050
LABELS = ['Speech', 'Baby cry, infant cry', 'Screaming', 'Slap, smack', 'Silence']


def tag_timeline(duration, cry=(), speech=(), scream=(), impact=()):
    """1 s hop, 3 s windows; each kwarg is a list of (t0, t1, prob) spans."""
    starts = np.arange(0, max(1, int(duration) - 2), 1.0)
    probs = np.zeros((len(starts), len(LABELS)), dtype=np.float32)

    def paint(spans, label):
        for t0, t1, p in spans:
            for i, s in enumerate(starts):
                if s + 3.0 > t0 and s < t1:
                    probs[i, LABELS.index(label)] = max(probs[i, LABELS.index(label)], p)
    paint(cry, 'Baby cry, infant cry')
    paint(speech, 'Speech')
    paint(scream, 'Screaming')
    paint(impact, 'Slap, smack')
    return TagTimeline(starts=starts, window=3.0, probs=probs, labels=LABELS, duration=duration)


def emotion_timeline(duration, spans):
    """spans: list of (t0, t1, arousal, dominance, valence)."""
    starts = np.arange(0, max(1, int(duration) - 4), 2.5)
    a = np.full(len(starts), 0.3, dtype=np.float32)
    d = np.full(len(starts), 0.4, dtype=np.float32)
    v = np.full(len(starts), 0.5, dtype=np.float32)
    for t0, t1, ar, do, va in spans:
        for i, s in enumerate(starts):
            if s + 5.0 > t0 and s < t1:
                a[i], d[i], v[i] = ar, do, va
    return EmotionTimeline(starts=starts, window=5.0, duration=duration, arousal=a, dominance=d,
                           valence=v, categorical={'ang': np.zeros(len(starts), dtype=np.float32)})


def tone(duration, amp=0.2, freq=220):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestCryDetectorML(unittest.TestCase):

    def setUp(self):
        self.det = CryDetector()
        self.audio = tone(40)

    def test_cry_segments_from_timeline(self):
        thr = config.tagger.cry_threshold
        tl = tag_timeline(40, cry=[(10, 16, thr + 0.3)])
        segs = self.det.detect_cry_segments(self.audio, SR, timeline=tl)
        self.assertEqual(len(segs), 1)
        seg = segs[0]
        self.assertTrue(seg['ml_backed'])
        self.assertLessEqual(seg['start_time'], 10.0)
        self.assertGreaterEqual(seg['end_time'], 16.0)
        self.assertAlmostEqual(seg['confidence'], thr + 0.3, places=4)
        self.assertIn(seg['intensity'], ('low', 'medium', 'high'))
        self.assertIn('cry_score', seg['features'])

    def test_no_cry_below_threshold(self):
        tl = tag_timeline(40, cry=[(10, 16, config.tagger.cry_threshold / 2)])
        self.assertEqual(self.det.detect_cry_segments(self.audio, SR, timeline=tl), [])

    def test_response_detected_from_speech_after_cry(self):
        thr = config.tagger.cry_threshold
        tl = tag_timeline(40, cry=[(10, 14, thr + 0.3)], speech=[(15, 19, 0.95)])
        segs = self.det.detect_cry_segments(self.audio, SR, timeline=tl)
        with_resp = self.det.detect_response_to_cry(self.audio, SR, segs, timeline=tl)
        self.assertTrue(with_resp[0]['response_detected'])
        self.assertIn(with_resp[0]['response_quality'], ('poor', 'adequate', 'good'))
        self.assertGreaterEqual(with_resp[0]['response_start'], segs[0]['end_time'])

    def test_crying_is_not_counted_as_a_response(self):
        thr = config.tagger.cry_threshold
        # continuous crying also lights up the AudioSet speech label a little
        tl = tag_timeline(40, cry=[(10, 30, thr + 0.5)], speech=[(10, 30, 0.7)])
        segs = self.det.detect_cry_segments(self.audio, SR, timeline=tl)
        with_resp = self.det.detect_response_to_cry(self.audio, SR, segs, timeline=tl)
        self.assertFalse(with_resp[0]['response_detected'])

    def test_measure_response_time_with_timeline(self):
        thr = config.tagger.cry_threshold
        tl = tag_timeline(40, cry=[(5, 9, thr + 0.3)], speech=[(20, 24, 0.95)])
        segs = self.det.detect_cry_segments(self.audio, SR, timeline=tl)
        enriched = self.det.measure_response_time(self.audio, SR, segs, timeline=tl)
        self.assertIsNotNone(enriched[0]['response_time_seconds'])
        self.assertEqual(enriched[0]['response_rating'], 'immediate')


class TestViolenceDetectorML(unittest.TestCase):

    def setUp(self):
        self.det = ViolenceDetector()
        self.audio = tone(40)

    def test_scream_becomes_shouting(self):
        tl = tag_timeline(40, scream=[(20, 23, config.tagger.scream_threshold + 0.2)])
        segs = self.det.detect_violence_segments(self.audio, SR, timeline=tl)
        self.assertEqual(len(segs), 1)
        self.assertIn('shouting', segs[0]['violence_types'])
        self.assertTrue(segs[0]['ml_backed'])
        self.assertIn(segs[0]['severity'], ('low', 'medium', 'high', 'critical'))
        self.assertIn('overall_assessment', segs[0]['context'])

    def test_impact_becomes_physical_indicator(self):
        tl = tag_timeline(40, impact=[(20, 21, config.tagger.impact_threshold + 0.2)])
        segs = self.det.detect_violence_segments(self.audio, SR, timeline=tl)
        self.assertEqual(len(segs), 1)
        self.assertIn('potential_physical_violence', segs[0]['violence_types'])

    def test_aggressive_tone_requires_dominant_speech(self):
        vc = config.violence
        emo = emotion_timeline(40, [(10, 20, vc.aggressive_arousal + 0.2, vc.aggressive_dominance + 0.1,
                                     vc.aggressive_valence_max - 0.1)])
        # no speech in the tagger -> gated out
        tl = tag_timeline(40)
        self.assertEqual(self.det.detect_violence_segments(self.audio, SR, timeline=tl, emotion_timeline=emo), [])
        # with speech -> aggressive tone
        tl = tag_timeline(40, speech=[(10, 20, 0.9)])
        segs = self.det.detect_violence_segments(self.audio, SR, timeline=tl, emotion_timeline=emo)
        self.assertTrue(segs)
        self.assertIn('aggressive_tone', segs[0]['violence_types'])
        # crying child with high arousal, speech not dominant -> not aggressive
        tl = tag_timeline(40, speech=[(10, 20, 0.7)], cry=[(10, 20, 0.9)])
        self.assertEqual(self.det.detect_violence_segments(self.audio, SR, timeline=tl, emotion_timeline=emo), [])

    def test_silent_timelines_give_nothing(self):
        tl = tag_timeline(40)
        emo = emotion_timeline(40, [])
        self.assertEqual(self.det.detect_violence_segments(self.audio, SR, timeline=tl, emotion_timeline=emo), [])


class TestNeglectDetectorML(unittest.TestCase):

    def setUp(self):
        self.det = NeglectDetector()
        self.audio = tone(120)

    def test_unanswered_cry_uses_tagger_speech(self):
        cry = [{'start_time': 10.0, 'end_time': 20.0, 'duration': 10.0, 'intensity': 'high'}]
        answered = tag_timeline(120, speech=[(21, 27, 0.95)])
        ignored = tag_timeline(120)
        self.assertEqual(len(self.det.detect_neglect_patterns(self.audio, SR, cry, [], timeline=answered)['unanswered_cries']), 0)
        self.assertEqual(len(self.det.detect_neglect_patterns(self.audio, SR, cry, [], timeline=ignored)['unanswered_cries']), 1)
        self.assertIsNone(self.det._timeline)   # reset after the call

    def test_adult_speech_ratio_from_timeline(self):
        tl = tag_timeline(120, speech=[(0, 60, 0.9)])
        self.det._timeline = tl
        try:
            ratio = self.det._calculate_adult_speech_ratio(self.audio, SR, t0=0.0)
        finally:
            self.det._timeline = None
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)


class TestAdvancedAnalyzerRules(unittest.TestCase):

    def test_concerning_emotions_from_timeline_gated_by_speech(self):
        an = AdvancedAnalyzer(use_whisper=False, use_transformer_emotion=False)
        cfg = config.advanced
        emo = emotion_timeline(40, [(10, 20, cfg.arousal_threshold + 0.2, cfg.dominance_threshold + 0.2,
                                     cfg.valence_max - 0.1)])
        self.assertEqual(an.concerning_emotions_from_timeline(emo, tag_timeline(40)), [])
        res = an.concerning_emotions_from_timeline(emo, tag_timeline(40, speech=[(10, 20, 0.9)]))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['detected_emotion'], 'aggression')
        self.assertTrue(res[0]['ml_backed'])
        self.assertIn(res[0]['severity'], ('low', 'medium', 'high', 'critical'))
        self.assertLessEqual(res[0]['start_time'], 10.0)
        self.assertGreaterEqual(res[0]['end_time'], 20.0)

    def test_transcription_shared_with_language_detector(self):
        det = InappropriateLanguageDetector()
        shared = {'text': 'you are so stupid and worthless', 'segments': [
            {'start': 0.0, 'end': 2.5, 'text': 'you are so stupid and worthless'}]}
        out = det.analyze_with_whisper('/nonexistent.wav', language='en', transcription=shared)
        self.assertNotIn('status', out)
        self.assertEqual(out['transcription'], shared['text'])
        self.assertGreaterEqual(out['detected_inappropriate_words'], 1)
        self.assertTrue(out['has_inappropriate_language'])


if __name__ == '__main__':
    unittest.main()
