# Real-audio evaluation suite

Everything in `evaluation/` measures the **production detector code paths** on
public, labelled audio.  Nothing here is synthetic.

## Datasets (downloaded to `data/`, git-ignored)

| dataset | what | used as | licence |
|---|---|---|---|
| ESC-50 | 2000 × 5 s environmental clips, 50 classes | 40 `crying_baby` positives; everything else as negatives (incl. laughing, coughing, sneezing, door knocks, glass breaking) | CC BY-NC |
| Donate-a-Cry | 457 real infant cries (phone recordings) | cry positives | CC BY-SA |
| RAVDESS (speech) | 1440 acted English sentences, 8 emotions × 2 intensities | angry positives; neutral/calm/happy negatives; strong anger for violence | CC BY-NC-SA |
| CREMA-D (test+val) | 2233 acted English sentences, 6 emotions, 91 actors | anger positives; neutral/happy negatives | ODbL |
| VIVAE | 1085 non-verbal vocalisations, 6 affects × 4 intensities | strong/peak anger, fear, pain = scream positives | CC BY |
| LibriSpeech dev-clean | read audiobook speech | calm adult-speech negatives; English ASR reference | CC BY |
| FLEURS he_il dev | 328 read Hebrew sentences with transcripts | Hebrew ASR benchmark | CC BY |

Clips are split **dev / test by source** (ESC-50 folds 1-3 vs 4-5, RAVDESS actors
1-12 vs 13-24, VIVAE speakers 1-6 vs 7-11, hashed speaker ids for CREMA-D and
LibriSpeech, hashed file ids for Donate-a-Cry).  Thresholds are chosen on *dev*
(`evaluation/fusion.py`) and every number quoted in the docs is from *test*.

## Tasks

* **cry** — does the clip contain an infant cry?  (497 pos / 2974 neg)
* **violence** — vocal aggression: screams/shouts or strong angry speech? (371 / 3534)
* **anger** — angry adult speech vs neutral, calm, happy speech? (573 / 1587)
* **speech** — is adult speech present? (the staff-response signal; 2161 / 2057)
* **asr_he / asr_en** — word error rate of the Whisper models

## Commands

```bash
.venv/bin/pip install -r requirements-eval.txt
bash evaluation/download_datasets.sh          # ~2.3 GB into data/ (git-ignored)
.venv/bin/python -m evaluation.run_eval --tag final --backends heuristic,hubert,tagger,emotion2,fusion
.venv/bin/python -m evaluation.run_eval --tag asr --tasks asr_he,asr_en
.venv/bin/python -m evaluation.fusion            # dev-tuned thresholds + fusion rules
.venv/bin/python -m evaluation.montage --minutes 20   # long-recording test through main.py
```

Runner outputs are cached in `data/cache/*.jsonl`; reports are written to
`evaluation/results/<tag>.{json,md}`.

## Backends

* `heuristic` — the spectral-threshold detectors (the v2.x primary path)
* `hubert` — `superb/hubert-large-superb-er` categorical emotion (v2.x "ML primary")
* `tagger` — AudioSet tagger (`audio_tagger.py`)
* `emotion2` — dimensional + categorical emotion (`emotion_models.py`)
* `fusion` — the v3 production rules over `tagger` + `emotion2`

## Montage test

`montage.py` concatenates *test-split* clips into 20-minute recordings (with and
without inserted cry / scream / anger events) and runs `main.py`'s pipeline on
them.  It reports event recall and **false alarms per hour**, the number an
operator actually experiences.  Its negatives are deliberately hard (acted,
theatrical "happy" and "surprised" speech, animals, laughter), so real-world
false-alarm rates are expected to be lower.
