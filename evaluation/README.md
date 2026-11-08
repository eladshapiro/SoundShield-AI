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
| FLEURS he_il dev | 328 read Hebrew sentences with transcripts | Hebrew ASR benchmark; Hebrew-speech negatives for the heads | CC BY |
| UrbanSound8K `children_playing` | 637 × 4 s slices of real kids at play (112 recordings) | negatives for every detector; "kindergarten" montage background | CC BY-NC |

Clips are split **dev / test by source** (ESC-50 folds 1-3 vs 4-5, RAVDESS actors
1-12 vs 13-24, VIVAE speakers 1-6 vs 7-11, UrbanSound8K folds 1-5 vs 6-10, hashed
speaker ids for CREMA-D and LibriSpeech, hashed file ids for Donate-a-Cry).
Thresholds are chosen on *dev* (`evaluation/fusion.py`, `evaluation/heads.py`) and
every number quoted in the docs is from *test*.

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
.venv/bin/python -m evaluation.fetch_urbansound_children   # ~0.6 GB, kids-at-play clips
.venv/bin/python -m evaluation.run_eval --tag final --backends heuristic,hubert,tagger,emotion2,fusion
.venv/bin/python -m evaluation.run_eval --tag asr --tasks asr_he,asr_en
.venv/bin/python -m evaluation.fusion            # dev-tuned thresholds + fusion rules (v3.0)
.venv/bin/python -m evaluation.heads extract     # v3.1 heads: embeddings for clips + montages (GPU, ~15 min, cached)
.venv/bin/python -m evaluation.heads train       # fit on dev, write heads/anger.json + heads/scream.json
.venv/bin/python -m evaluation.heads report      # test split + test montages -> results/heads.md
.venv/bin/python -m evaluation.montage --minutes 20                   # long-recording test through main.py
.venv/bin/python -m evaluation.montage --minutes 20 --kids-share 0.5  # same, "kindergarten" background
```

Runner outputs are cached in `data/cache/*.jsonl`; reports are written to
`evaluation/results/<tag>.{json,md}`.

## Backends

* `heuristic` — the spectral-threshold detectors (the v2.x primary path)
* `hubert` — `superb/hubert-large-superb-er` categorical emotion (v2.x "ML primary")
* `tagger` — AudioSet tagger (`audio_tagger.py`)
* `emotion2` — dimensional + categorical emotion (`emotion_models.py`)
* `fusion` — the v3.0 production rules over `tagger` + `emotion2`
* learned heads (v3.1) are scored by `evaluation.heads report`, not `run_eval`

## Montage test

`montage.py` concatenates *test-split* clips into 20-minute recordings (with and
without inserted cry / scream / anger events) and runs `main.py`'s pipeline on
them.  It reports event recall and **false alarms per hour**, the number an
operator actually experiences.  Its negatives are deliberately hard (acted,
theatrical "happy" and "surprised" speech, animals, laughter).  `--kids-share 0.5`
makes half of the background pieces real kids at play (UrbanSound8K), the most
common sound in an actual kindergarten and the one the v3.0 `shouting` rule
mistook for screaming 84-108 times per hour.

## Learned heads (v3.1)

`heads.py` trains the small logistic-regression heads that `event_heads.py` runs
on top of the frozen models' embeddings (wav2vec2 MSP-Dim, HuBERT SUPERB-ER, AST):

* `anger`  — angry adult speech (concerning emotion + `aggressive_tone`)
* `scream` — distress screams (fear / pain / anger) vs joyful screams, kids at play, cries (`shouting`)

Protocol: the dev split is cut into two speaker folds; each fold gets its own
20-minute montages (plain and kindergarten) built only from that fold's clips, so
cross-fitted predictions never come from a head that heard the same clip.
Feature blocks and regularisation are chosen by out-of-fold average precision,
the threshold by the lowest false-alarm rate per hour that keeps 80% event recall
on the dev montages, and the shipped head is refitted on all dev data.  The test
split and the test montages (the same seeds 0/1 as the v3.0 report, plus
kindergarten versions) are only scored, with the v3.0 rules run on the identical
windows through the same detector code (`results/heads.md`).  Embeddings are cached
in `data/cache/heads/`.
