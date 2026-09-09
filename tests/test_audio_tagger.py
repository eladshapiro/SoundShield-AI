"""Unit tests for audio_tagger.TagTimeline / AudioTagger (no model download needed)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_tagger import AudioTagger, LABEL_GROUPS, TagTimeline  # noqa: E402
from config import config  # noqa: E402


LABELS = ['Speech', 'Baby cry, infant cry', 'Crying, sobbing', 'Screaming', 'Shout',
          'Laughter', 'Slap, smack', 'Silence', 'Music', 'Child speech, kid speaking']


def make_timeline(rows, window=3.0, hop=1.0):
    """rows: list of dicts label -> prob, one per window."""
    probs = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for i, r in enumerate(rows):
        for k, v in r.items():
            probs[i, LABELS.index(k)] = v
    starts = np.arange(len(rows)) * hop
    duration = starts[-1] + window if len(rows) else 0.0
    return TagTimeline(starts=starts, window=window, probs=probs, labels=LABELS, duration=duration)


class TestTagTimeline(unittest.TestCase):

    def test_group_scores_take_max_over_labels(self):
        tl = make_timeline([{'Baby cry, infant cry': 0.2, 'Crying, sobbing': 0.6}])
        self.assertAlmostEqual(float(tl.group_scores('cry')[0]), 0.6)
        self.assertAlmostEqual(float(tl.group_scores('infant_cry')[0]), 0.2)
        self.assertAlmostEqual(float(tl.group_scores('scream')[0]), 0.0)

    def test_unknown_group_is_zero(self):
        tl = make_timeline([{'Speech': 0.9}])
        self.assertEqual(float(tl.group_scores('no_such_group')[0]), 0.0)

    def test_score_reduces_over_time_range(self):
        tl = make_timeline([{'Speech': 0.1}, {'Speech': 0.9}, {'Speech': 0.3}])
        self.assertAlmostEqual(tl.score('speech', 0, 1), 0.1)       # only window 0 overlaps [0,1)
        self.assertAlmostEqual(tl.score('speech', 1, 2), 0.9)       # windows 0 and 1 overlap
        self.assertAlmostEqual(tl.score('speech', 0, None), 0.9)
        self.assertAlmostEqual(tl.score('speech', 0, None, reduce='mean'), (0.1 + 0.9 + 0.3) / 3, places=5)
        self.assertEqual(tl.score('speech', 100, 200), 0.0)

    def test_segments_are_contiguous_runs(self):
        # 3 s windows, 1 s hop: hits at windows 0-1 span [0, 4); window 5 spans [5, 8)
        tl = make_timeline([{'Screaming': 0.9}, {'Screaming': 0.8}, {}, {}, {}, {'Shout': 0.5}])
        segs = tl.segments('scream', threshold=0.5)
        self.assertEqual(len(segs), 2)
        start, end, peak = segs[0]
        self.assertEqual(start, 0.0)
        self.assertEqual(end, 1.0 + 3.0)
        self.assertAlmostEqual(peak, 0.9)
        self.assertEqual(segs[1][0], 5.0)

    def test_touching_runs_merge(self):
        # windows overlap in time, so a single missed window does not split an event
        tl = make_timeline([{'Screaming': 0.9}, {}, {'Screaming': 0.9}, {'Screaming': 0.9}])
        self.assertEqual(len(tl.segments('scream', 0.5, merge_gap=0.0)), 1)

    def test_segments_merge_gap_and_min_windows(self):
        tl = make_timeline([{'Screaming': 0.9}, {}, {}, {}, {'Screaming': 0.9}, {'Screaming': 0.9}])
        self.assertEqual(len(tl.segments('scream', 0.5, merge_gap=0.0)), 2)
        self.assertEqual(len(tl.segments('scream', 0.5, merge_gap=5.0)), 1)
        segs = tl.segments('scream', 0.5, min_windows=2)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], 4.0)

    def test_speech_dominant_requires_margin_over_cry(self):
        margin = config.tagger.speech_over_cry_margin
        tl = make_timeline([
            {'Speech': 0.9, 'Baby cry, infant cry': 0.1},                 # speech dominates
            {'Speech': 0.9, 'Baby cry, infant cry': 0.9 - margin / 2},    # within margin -> 0
            {'Speech': 0.5, 'Crying, sobbing': 0.8},                      # cry dominates -> 0
        ])
        sd = tl.group_scores('speech_dominant')
        self.assertAlmostEqual(float(sd[0]), 0.9)
        self.assertEqual(float(sd[1]), 0.0)
        self.assertEqual(float(sd[2]), 0.0)

    def test_top_labels_and_to_dict(self):
        tl = make_timeline([{'Speech': 0.9, 'Laughter': 0.4}, {'Music': 0.7}])
        top = tl.top_labels(0, 1, k=2)
        self.assertEqual(top[0][0], 'Speech')
        d = tl.to_dict()
        self.assertEqual(len(d['starts']), 2)
        self.assertIn('cry', d['groups'])
        self.assertIn('speech_dominant', d['groups'])

    def test_label_groups_reference_real_audioset_names(self):
        # Every configured group label must be a plausible AudioSet display name (non-empty, capitalised)
        for group, names in LABEL_GROUPS.items():
            self.assertTrue(names, group)
            for n in names:
                self.assertTrue(n[0].isupper(), n)


class TestAudioTaggerWindows(unittest.TestCase):

    def test_windows_cover_the_tail(self):
        tagger = AudioTagger(window=3.0, hop=1.0)
        starts = tagger.windows(int(16000 * 10.5))
        self.assertEqual(starts[0], 0.0)
        self.assertAlmostEqual(starts[-1], 7.5, places=3)          # last window ends at 10.5 s
        self.assertTrue(np.all(np.diff(starts[:-1]) >= 0.999))

    def test_short_audio_gets_single_window(self):
        tagger = AudioTagger(window=3.0, hop=1.0)
        self.assertEqual(list(tagger.windows(16000)), [0.0])

    def test_unavailable_model_returns_none_without_raising(self):
        previous = os.environ.get('HF_HUB_OFFLINE')
        os.environ['HF_HUB_OFFLINE'] = '1'          # never hit the network for a bogus repo
        try:
            tagger = AudioTagger(model_name='definitely/not-a-real-model-xyz')
            self.assertIsNone(tagger.tag(np.zeros(16000, dtype=np.float32), 16000))
            self.assertFalse(tagger.loaded)
            self.assertIsNotNone(tagger.load_error)
        finally:
            if previous is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = previous


if __name__ == '__main__':
    unittest.main()
