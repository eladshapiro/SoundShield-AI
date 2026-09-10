# Evaluation report — `final`

Generated 2026-09-09 17:30 · config hash `6824ca6104`

## Task: cry

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | test F1 @ dev-tuned thr |
|---|---|---|---|---|---|---|---|---|
| heuristic | 497 | 2974 | 0.166 | 0.907 | 0.281 | 0.760 | 0.728 | 0.436 @ 0.367 |
| tagger | 497 | 2974 | 0.976 | 0.887 | 0.929 | 0.004 | 0.996 | 0.927 @ 0.065 |
| fusion | 497 | 2974 | 0.976 | 0.887 | 0.929 | 0.004 | 0.996 | 0.927 @ 0.065 |

<details><summary>cry / heuristic: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/neutral | 326 | 0 | FPR=0.629 |
| donateacry/infant_cry | 457 | 1 | recall=0.899 |
| esc50/animals | 400 | 0 | FPR=0.963 |
| esc50/domestic | 400 | 0 | FPR=0.745 |
| esc50/human_nonspeech | 400 | 0 | FPR=0.807 |
| esc50/natural | 400 | 0 | FPR=0.785 |
| esc50/urban | 400 | 0 | FPR=0.970 |
| librispeech/read_speech | 400 | 0 | FPR=0.955 |
| ravdess/calm | 192 | 0 | FPR=0.016 |
| ravdess/neutral | 96 | 0 | FPR=0.031 |

</details>

<details><summary>cry / tagger: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/neutral | 326 | 0 | FPR=0.000 |
| donateacry/infant_cry | 457 | 1 | recall=0.877 |
| esc50/animals | 400 | 0 | FPR=0.005 |
| esc50/domestic | 400 | 0 | FPR=0.005 |
| esc50/human_nonspeech | 400 | 0 | FPR=0.115 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 0 | FPR=0.003 |
| ravdess/calm | 192 | 0 | FPR=0.000 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |

</details>

<details><summary>cry / fusion: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/neutral | 326 | 0 | FPR=0.000 |
| donateacry/infant_cry | 457 | 1 | recall=0.877 |
| esc50/animals | 400 | 0 | FPR=0.005 |
| esc50/domestic | 400 | 0 | FPR=0.005 |
| esc50/human_nonspeech | 400 | 0 | FPR=0.115 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 0 | FPR=0.003 |
| ravdess/calm | 192 | 0 | FPR=0.000 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |

</details>

## Task: violence

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | test F1 @ dev-tuned thr |
|---|---|---|---|---|---|---|---|---|
| heuristic | 371 | 3534 | 0.058 | 0.132 | 0.081 | 0.225 | 0.447 | 0.168 @ 0.000 |
| tagger | 371 | 3534 | 0.741 | 0.642 | 0.688 | 0.023 | 0.956 | 0.652 @ 0.008 |
| emotion2 | 371 | 3534 | 0.540 | 0.275 | 0.364 | 0.025 | 0.626 | 0.383 @ 0.733 |
| fusion | 371 | 3534 | 0.772 | 0.814 | 0.793 | 0.025 | 0.969 | 0.758 @ 0.009 |

<details><summary>violence / heuristic: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 0 | FPR=0.055 |
| cremad/neutral | 326 | 0 | FPR=0.000 |
| esc50/animals | 400 | 0 | FPR=0.472 |
| esc50/domestic | 400 | 0 | FPR=0.273 |
| esc50/human_nonspeech | 360 | 0 | FPR=0.217 |
| esc50/natural | 400 | 0 | FPR=0.307 |
| esc50/urban | 400 | 0 | FPR=0.610 |
| librispeech/read_speech | 400 | 0 | FPR=0.075 |
| ravdess/angry | 96 | 1 | recall=0.302 |
| ravdess/calm | 96 | 0 | FPR=0.000 |
| ravdess/happy | 96 | 0 | FPR=0.000 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |
| vivae/achievement | 83 | 0 | FPR=0.012 |
| vivae/anger | 88 | 1 | recall=0.011 |
| vivae/fear | 91 | 1 | recall=0.121 |
| vivae/pain | 96 | 1 | recall=0.083 |
| vivae/pleasure | 96 | 0 | FPR=0.000 |

</details>

<details><summary>violence / tagger: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 0 | FPR=0.003 |
| cremad/neutral | 326 | 0 | FPR=0.000 |
| esc50/animals | 400 | 0 | FPR=0.070 |
| esc50/domestic | 400 | 0 | FPR=0.013 |
| esc50/human_nonspeech | 360 | 0 | FPR=0.022 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 0 | FPR=0.000 |
| ravdess/angry | 96 | 1 | recall=0.312 |
| ravdess/calm | 96 | 0 | FPR=0.000 |
| ravdess/happy | 96 | 0 | FPR=0.000 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |
| vivae/achievement | 83 | 0 | FPR=0.446 |
| vivae/anger | 88 | 1 | recall=0.841 |
| vivae/fear | 91 | 1 | recall=0.780 |
| vivae/pain | 96 | 1 | recall=0.656 |
| vivae/pleasure | 96 | 0 | FPR=0.042 |

</details>

<details><summary>violence / emotion2: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 0 | FPR=0.003 |
| cremad/neutral | 326 | 0 | FPR=0.000 |
| esc50/animals | 400 | 0 | FPR=0.058 |
| esc50/domestic | 400 | 0 | FPR=0.030 |
| esc50/human_nonspeech | 360 | 0 | FPR=0.039 |
| esc50/natural | 400 | 0 | FPR=0.020 |
| esc50/urban | 400 | 0 | FPR=0.058 |
| librispeech/read_speech | 400 | 0 | FPR=0.005 |
| ravdess/angry | 96 | 1 | recall=0.969 |
| ravdess/calm | 96 | 0 | FPR=0.000 |
| ravdess/happy | 96 | 0 | FPR=0.031 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |
| vivae/achievement | 83 | 0 | FPR=0.012 |
| vivae/anger | 88 | 1 | recall=0.045 |
| vivae/fear | 91 | 1 | recall=0.044 |
| vivae/pain | 96 | 1 | recall=0.010 |
| vivae/pleasure | 96 | 0 | FPR=0.000 |

</details>

<details><summary>violence / fusion: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 0 | FPR=0.005 |
| cremad/neutral | 326 | 0 | FPR=0.000 |
| esc50/animals | 400 | 0 | FPR=0.070 |
| esc50/domestic | 400 | 0 | FPR=0.013 |
| esc50/human_nonspeech | 360 | 0 | FPR=0.022 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 0 | FPR=0.005 |
| ravdess/angry | 96 | 1 | recall=0.969 |
| ravdess/calm | 96 | 0 | FPR=0.000 |
| ravdess/happy | 96 | 0 | FPR=0.031 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |
| vivae/achievement | 83 | 0 | FPR=0.446 |
| vivae/anger | 88 | 1 | recall=0.852 |
| vivae/fear | 91 | 1 | recall=0.780 |
| vivae/pain | 96 | 1 | recall=0.656 |
| vivae/pleasure | 96 | 0 | FPR=0.042 |

</details>

## Task: speech

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | test F1 @ dev-tuned thr |
|---|---|---|---|---|---|---|---|---|
| heuristic | 2161 | 2057 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.716 @ 0.000 |
| tagger | 2161 | 2057 | 0.966 | 0.988 | 0.977 | 0.037 | 0.978 | 0.977 @ 0.598 |
| fusion | 2161 | 2057 | 0.966 | 0.988 | 0.977 | 0.037 | 0.978 | 0.977 @ 0.598 |

<details><summary>speech / heuristic: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 1 | recall=0.000 |
| cremad/neutral | 326 | 1 | recall=0.000 |
| cremad/sad | 382 | 1 | recall=0.000 |
| donateacry/infant_cry | 457 | 0 | FPR=0.000 |
| esc50/animals | 400 | 0 | FPR=0.000 |
| esc50/domestic | 400 | 0 | FPR=0.000 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 1 | recall=0.000 |
| ravdess/calm | 192 | 1 | recall=0.000 |
| ravdess/happy | 192 | 1 | recall=0.000 |
| ravdess/neutral | 96 | 1 | recall=0.000 |
| ravdess/sad | 192 | 1 | recall=0.000 |

</details>

<details><summary>speech / tagger: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 1 | recall=0.997 |
| cremad/neutral | 326 | 1 | recall=0.997 |
| cremad/sad | 382 | 1 | recall=1.000 |
| donateacry/infant_cry | 457 | 0 | FPR=0.164 |
| esc50/animals | 400 | 0 | FPR=0.003 |
| esc50/domestic | 400 | 0 | FPR=0.000 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 1 | recall=0.968 |
| ravdess/calm | 192 | 1 | recall=0.958 |
| ravdess/happy | 192 | 1 | recall=1.000 |
| ravdess/neutral | 96 | 1 | recall=1.000 |
| ravdess/sad | 192 | 1 | recall=0.984 |

</details>

<details><summary>speech / fusion: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/happy | 381 | 1 | recall=0.997 |
| cremad/neutral | 326 | 1 | recall=0.997 |
| cremad/sad | 382 | 1 | recall=1.000 |
| donateacry/infant_cry | 457 | 0 | FPR=0.164 |
| esc50/animals | 400 | 0 | FPR=0.003 |
| esc50/domestic | 400 | 0 | FPR=0.000 |
| esc50/natural | 400 | 0 | FPR=0.000 |
| esc50/urban | 400 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 1 | recall=0.968 |
| ravdess/calm | 192 | 1 | recall=0.958 |
| ravdess/happy | 192 | 1 | recall=1.000 |
| ravdess/neutral | 96 | 1 | recall=1.000 |
| ravdess/sad | 192 | 1 | recall=0.984 |

</details>

## Task: anger

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | test F1 @ dev-tuned thr |
|---|---|---|---|---|---|---|---|---|
| heuristic | 573 | 1587 | 0.590 | 0.080 | 0.141 | 0.020 | 0.530 | 0.407 @ 0.000 |
| hubert | 573 | 1587 | 0.514 | 0.876 | 0.648 | 0.299 | 0.857 | 0.644 @ 0.778 |
| emotion2 | 573 | 1587 | 0.680 | 0.686 | 0.683 | 0.117 | 0.806 | 0.661 @ 0.795 |
| fusion | 573 | 1587 | 0.679 | 0.681 | 0.680 | 0.116 | 0.803 | 0.656 @ 0.795 |

<details><summary>anger / heuristic: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/anger | 381 | 1 | recall=0.115 |
| cremad/happy | 381 | 0 | FPR=0.010 |
| cremad/neutral | 326 | 0 | FPR=0.000 |
| librispeech/read_speech | 400 | 0 | FPR=0.070 |
| ravdess/angry | 192 | 1 | recall=0.010 |
| ravdess/calm | 192 | 0 | FPR=0.000 |
| ravdess/happy | 192 | 0 | FPR=0.000 |
| ravdess/neutral | 96 | 0 | FPR=0.000 |

</details>

<details><summary>anger / hubert: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/anger | 381 | 1 | recall=0.819 |
| cremad/happy | 381 | 0 | FPR=0.142 |
| cremad/neutral | 326 | 0 | FPR=0.015 |
| librispeech/read_speech | 400 | 0 | FPR=0.175 |
| ravdess/angry | 192 | 1 | recall=0.990 |
| ravdess/calm | 192 | 0 | FPR=0.557 |
| ravdess/happy | 192 | 0 | FPR=0.807 |
| ravdess/neutral | 96 | 0 | FPR=0.865 |

</details>

<details><summary>anger / emotion2: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/anger | 381 | 1 | recall=0.562 |
| cremad/happy | 381 | 0 | FPR=0.055 |
| cremad/neutral | 326 | 0 | FPR=0.003 |
| librispeech/read_speech | 400 | 0 | FPR=0.070 |
| ravdess/angry | 192 | 1 | recall=0.932 |
| ravdess/calm | 192 | 0 | FPR=0.005 |
| ravdess/happy | 192 | 0 | FPR=0.615 |
| ravdess/neutral | 96 | 0 | FPR=0.167 |

</details>

<details><summary>anger / fusion: per-group rates</summary>

| group | n | label | rate |
|---|---|---|---|
| cremad/anger | 381 | 1 | recall=0.556 |
| cremad/happy | 381 | 0 | FPR=0.055 |
| cremad/neutral | 326 | 0 | FPR=0.003 |
| librispeech/read_speech | 400 | 0 | FPR=0.068 |
| ravdess/angry | 192 | 1 | recall=0.927 |
| ravdess/calm | 192 | 0 | FPR=0.005 |
| ravdess/happy | 192 | 0 | FPR=0.615 |
| ravdess/neutral | 96 | 0 | FPR=0.167 |

</details>
