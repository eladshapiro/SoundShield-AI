# Evaluation report — `baseline`

Generated 2026-09-09 16:48 · config hash `4407b47de0`

## Task: cry

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | best-thr F1 |
|---|---|---|---|---|---|---|---|---|
| heuristic | 497 | 2974 | 0.166 | 0.907 | 0.281 | 0.760 | 0.728 | 0.433 @ 0.367 |

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

## Task: violence

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | best-thr F1 |
|---|---|---|---|---|---|---|---|---|
| heuristic | 371 | 3534 | 0.058 | 0.132 | 0.081 | 0.225 | 0.447 | 0.174 @ 0.000 |

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

## Task: anger

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | best-thr F1 |
|---|---|---|---|---|---|---|---|---|
| heuristic | 573 | 1587 | 0.590 | 0.080 | 0.141 | 0.020 | 0.530 | 0.419 @ 0.000 |
| hubert | 573 | 1587 | 0.514 | 0.876 | 0.648 | 0.299 | 0.857 | 0.680 @ 0.868 |

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

## Task: speech

| backend | pos | neg | precision | recall | F1 | FPR | ROC-AUC | best-thr F1 |
|---|---|---|---|---|---|---|---|---|
| heuristic | 2161 | 2057 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.678 @ 0.000 |

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
