# Learned decision heads — held-out results

Heads are fitted and their thresholds chosen on the **dev** split only (two speaker folds, cross-fitted, own montages per fold; `python -m evaluation.heads train`). Everything below is the **test split** and the test montages, scored through the production detector code.

* `heads/anger.json`: blocks `w2v2+hubert+dims+cat`, threshold 0.825 (dev montages: 8.4 false/h at recall 0.81), 10668 dev windows, OOF AP 0.814
* `heads/scream.json`: blocks `w2v2+hubert+ast+dims+cat+tags`, threshold 0.775 (dev montages: 1.9 false/h at recall 0.83), 11028 dev windows, OOF AP 0.715

## Angry speech, the v3.0 anger task (test clips, speech-gated, any window)

Same clips as `final.md`: RAVDESS / CREMA-D angry vs neutral, calm, happy, and LibriSpeech. v3.0 rule: `anger_rule`

| decision | pos | neg | precision | recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| v3.0 rules | 281 | 818 | 0.629 | 0.676 | 0.652 | 0.137 | 0.794 |
| learned head | 281 | 818 | 0.914 | 0.797 | 0.852 | 0.026 | 0.956 |

| group | n | rate | v3.0 rules | learned head |
|---|---|---|---|---|
| cremad/anger | 185 | recall | 0.541 | 0.800 |
| cremad/happy | 194 | FPR | 0.088 | 0.088 |
| cremad/neutral | 147 | FPR | 0.007 | 0.020 |
| librispeech/read_speech | 237 | FPR | 0.089 | 0.000 |
| ravdess/angry | 96 | recall | 0.938 | 0.792 |
| ravdess/calm | 96 | FPR | 0.010 | 0.000 |
| ravdess/happy | 96 | FPR | 0.688 | 0.010 |
| ravdess/neutral | 48 | FPR | 0.125 | 0.000 |

## Angry speech, every test clip (speech-gated, any window)

All other emotions, cries, kids at play, everyday sounds and Hebrew speech as negatives too. v3.0 rule: `anger_rule`

| decision | pos | neg | precision | recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| v3.0 rules | 281 | 3406 | 0.358 | 0.676 | 0.469 | 0.100 | 0.803 |
| learned head | 281 | 3406 | 0.762 | 0.797 | 0.779 | 0.021 | 0.973 |

| group | n | rate | v3.0 rules | learned head |
|---|---|---|---|---|
| cremad/anger | 185 | recall | 0.541 | 0.800 |
| cremad/disgust | 166 | FPR | 0.042 | 0.127 |
| cremad/fear | 168 | FPR | 0.137 | 0.077 |
| cremad/happy | 194 | FPR | 0.088 | 0.088 |
| cremad/neutral | 147 | FPR | 0.007 | 0.020 |
| cremad/sad | 185 | FPR | 0.000 | 0.016 |
| donateacry/infant_cry | 231 | FPR | 0.004 | 0.000 |
| esc50/animals | 160 | FPR | 0.000 | 0.000 |
| esc50/domestic | 160 | FPR | 0.000 | 0.000 |
| esc50/human_nonspeech | 160 | FPR | 0.000 | 0.000 |
| esc50/natural | 160 | FPR | 0.000 | 0.000 |
| esc50/urban | 160 | FPR | 0.000 | 0.000 |
| fleurs/he | 328 | FPR | 0.000 | 0.000 |
| librispeech/read_speech | 237 | FPR | 0.089 | 0.000 |
| ravdess/angry | 96 | recall | 0.938 | 0.792 |
| ravdess/calm | 96 | FPR | 0.010 | 0.000 |
| ravdess/disgust | 96 | FPR | 0.708 | 0.115 |
| ravdess/fearful | 96 | FPR | 0.417 | 0.010 |
| ravdess/happy | 96 | FPR | 0.688 | 0.010 |
| ravdess/neutral | 48 | FPR | 0.125 | 0.000 |
| ravdess/sad | 96 | FPR | 0.083 | 0.000 |
| ravdess/surprised | 96 | FPR | 0.833 | 0.000 |
| us8k/children_playing | 326 | FPR | 0.003 | 0.000 |

## Distress screams, test clips (any window)

v3.0 rule: AudioSet scream group >= `TAGGER_SCREAM_THRESHOLD`

| decision | pos | neg | precision | recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| v3.0 rules | 112 | 3882 | 0.259 | 0.732 | 0.383 | 0.060 | 0.917 |
| learned head | 112 | 3882 | 0.500 | 0.455 | 0.477 | 0.013 | 0.956 |

| group | n | rate | v3.0 rules | learned head |
|---|---|---|---|---|
| cremad/anger | 185 | FPR | 0.027 | 0.000 |
| cremad/disgust | 166 | FPR | 0.006 | 0.000 |
| cremad/fear | 168 | FPR | 0.006 | 0.000 |
| cremad/happy | 194 | FPR | 0.000 | 0.000 |
| cremad/neutral | 147 | FPR | 0.000 | 0.000 |
| cremad/sad | 185 | FPR | 0.000 | 0.000 |
| donateacry/infant_cry | 231 | FPR | 0.264 | 0.004 |
| esc50/animals | 160 | FPR | 0.069 | 0.000 |
| esc50/domestic | 160 | FPR | 0.019 | 0.000 |
| esc50/human_nonspeech | 160 | FPR | 0.062 | 0.013 |
| esc50/natural | 160 | FPR | 0.000 | 0.000 |
| esc50/urban | 160 | FPR | 0.000 | 0.000 |
| fleurs/he | 328 | FPR | 0.000 | 0.000 |
| librispeech/read_speech | 237 | FPR | 0.000 | 0.000 |
| ravdess/angry | 48 | FPR | 0.000 | 0.000 |
| ravdess/calm | 96 | FPR | 0.000 | 0.000 |
| ravdess/disgust | 96 | FPR | 0.000 | 0.000 |
| ravdess/fearful | 96 | FPR | 0.031 | 0.021 |
| ravdess/happy | 96 | FPR | 0.010 | 0.000 |
| ravdess/neutral | 48 | FPR | 0.000 | 0.000 |
| ravdess/sad | 96 | FPR | 0.000 | 0.000 |
| ravdess/surprised | 96 | FPR | 0.000 | 0.000 |
| us8k/children_playing | 326 | FPR | 0.110 | 0.000 |
| vivae/achievement | 72 | FPR | 0.625 | 0.167 |
| vivae/anger | 38 | recall | 0.763 | 0.632 |
| vivae/fear | 37 | recall | 0.757 | 0.297 |
| vivae/pain | 37 | recall | 0.676 | 0.432 |
| vivae/pleasure | 93 | FPR | 0.183 | 0.215 |
| vivae/surprise | 78 | FPR | 0.513 | 0.179 |

## 20-minute test montages (production detectors, heads off vs on)

`kg` = kindergarten montage: half the background pieces are UrbanSound8K kids at play.

| montage | detector | v3.0 rules: found / events | v3.0 false/h | heads: found / events | heads false/h |
|---|---|---|---|---|---|
| test_negative_0 | anger (concerning emotion) | - | 66.0 | - | 18.0 |
| test_negative_0 | shouting vs screams | - | 12.0 | - | 3.0 |
| test_negative_0 | aggressive tone vs anger | - | 12.0 | - | 18.0 |
| test_negative_0 | any violence vs screams+anger | - | 33.0 | - | 33.0 |
| test_events_0 | anger (concerning emotion) | 8 / 10 | 48.0 | 10 / 10 | 12.0 |
| test_events_0 | shouting vs screams | 6 / 8 | 30.0 | 6 / 8 | 12.0 |
| test_events_0 | aggressive tone vs anger | 5 / 10 | 12.0 | 10 / 10 | 12.0 |
| test_events_0 | any violence vs screams+anger | 11 / 18 | 39.0 | 16 / 18 | 27.0 |
| test_kg_negative_0 | anger (concerning emotion) | - | 84.0 | - | 12.0 |
| test_kg_negative_0 | shouting vs screams | - | 102.0 | - | 12.0 |
| test_kg_negative_0 | aggressive tone vs anger | - | 81.0 | - | 12.0 |
| test_kg_negative_0 | any violence vs screams+anger | - | 147.0 | - | 24.0 |
| test_kg_events_0 | anger (concerning emotion) | 7 / 9 | 57.0 | 9 / 9 | 9.0 |
| test_kg_events_0 | shouting vs screams | 8 / 10 | 93.0 | 7 / 10 | 9.0 |
| test_kg_events_0 | aggressive tone vs anger | 7 / 9 | 57.0 | 9 / 9 | 9.0 |
| test_kg_events_0 | any violence vs screams+anger | 16 / 19 | 93.0 | 16 / 19 | 18.0 |
| test_negative_1 | anger (concerning emotion) | - | 75.0 | - | 18.0 |
| test_negative_1 | shouting vs screams | - | 12.0 | - | 0.0 |
| test_negative_1 | aggressive tone vs anger | - | 18.0 | - | 12.0 |
| test_negative_1 | any violence vs screams+anger | - | 39.0 | - | 21.0 |
| test_events_1 | anger (concerning emotion) | 5 / 7 | 60.0 | 5 / 7 | 18.0 |
| test_events_1 | shouting vs screams | 5 / 9 | 21.0 | 7 / 9 | 0.0 |
| test_events_1 | aggressive tone vs anger | 1 / 7 | 15.0 | 5 / 7 | 18.0 |
| test_events_1 | any violence vs screams+anger | 7 / 16 | 39.0 | 12 / 16 | 24.0 |
| test_kg_negative_1 | anger (concerning emotion) | - | 81.0 | - | 9.0 |
| test_kg_negative_1 | shouting vs screams | - | 84.0 | - | 0.0 |
| test_kg_negative_1 | aggressive tone vs anger | - | 81.0 | - | 6.0 |
| test_kg_negative_1 | any violence vs screams+anger | - | 135.0 | - | 21.0 |
| test_kg_events_1 | anger (concerning emotion) | 5 / 6 | 39.0 | 4 / 6 | 9.0 |
| test_kg_events_1 | shouting vs screams | 6 / 10 | 108.0 | 6 / 10 | 3.0 |
| test_kg_events_1 | aggressive tone vs anger | 3 / 6 | 63.0 | 4 / 6 | 9.0 |
| test_kg_events_1 | any violence vs screams+anger | 11 / 16 | 120.0 | 10 / 16 | 21.0 |

**All test montages together** (8 × 20 min):

| detector | v3.0 recall | v3.0 false/h | heads recall | heads false/h |
|---|---|---|---|---|
| anger (concerning emotion) | 0.78 (25/32) | 63.8 | 0.88 (28/32) | 13.1 |
| shouting vs screams | 0.68 (25/37) | 57.8 | 0.70 (26/37) | 4.9 |
| aggressive tone vs anger | 0.50 (16/32) | 42.4 | 0.88 (28/32) | 12.0 |
| any violence vs screams+anger | 0.65 (45/69) | 80.6 | 0.78 (54/69) | 23.6 |
