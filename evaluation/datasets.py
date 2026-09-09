"""Dataset preparation and per-task manifests for the real-audio evaluation.

Every dataset is public.  Archives live in ``data/raw`` and are unpacked into
``data/<name>``.  ``build_task`` turns dataset indices into a list of
labelled clips for one detection task.

Datasets
--------
esc50        2000 x 5 s environmental clips (incl. 40 ``crying_baby``)         CC BY-NC
donateacry   457 real infant-cry phone recordings (7 s)                        CC BY-SA
ravdess      1440 acted English speech clips, 8 emotions x 2 intensities       CC BY-NC-SA
cremad       2234 acted English speech clips (test+validation), 6 emotions     ODbL
vivae        1085 non-verbal vocalisations, 6 affects x 4 intensities          CC BY
librispeech  dev-clean read speech (calm adult speech negatives)               CC BY
fleurs_he    328 Hebrew read sentences with transcripts (ASR benchmark)        CC BY
"""
from __future__ import annotations

import csv
import io
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'


@dataclass
class Clip:
    path: str
    dataset: str
    category: str          # dataset-native label (esc50 category, emotion, ...)
    group: str             # coarse group used for per-group breakdowns
    intensity: str = ''
    speaker: str = ''
    transcript: str = ''
    split: str = 'test'    # 'dev' (threshold tuning) or 'test' (reporting)

    def to_dict(self) -> Dict:
        return asdict(self)


def _hash_split(key: str, dev_fraction: float = 0.5) -> str:
    """Deterministic speaker/file-level split so tuning and test never share a source."""
    import hashlib
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000
    return 'dev' if h < dev_fraction * 1000 else 'test'


# ---------------------------------------------------------------------------
# Dataset indices
# ---------------------------------------------------------------------------
ESC50_GROUPS = ['animals', 'natural', 'human_nonspeech', 'domestic', 'urban']


def index_esc50() -> List[Clip]:
    meta = DATA_DIR / 'esc50' / 'meta' / 'esc50.csv'
    if not meta.exists():
        raise FileNotFoundError(f'ESC-50 not prepared: {meta}')
    clips = []
    with open(meta, newline='') as fh:
        for row in csv.DictReader(fh):
            target = int(row['target'])
            clips.append(Clip(
                path=str(DATA_DIR / 'esc50' / 'audio' / row['filename']),
                dataset='esc50',
                category=row['category'],
                group=f"esc50/{ESC50_GROUPS[target // 10]}",
                split='dev' if int(row['fold']) <= 3 else 'test',
            ))
    return clips


def index_donateacry() -> List[Clip]:
    base = DATA_DIR / 'donateacry-corpus-master' / 'donateacry_corpus_cleaned_and_updated_data'
    if not base.exists():
        raise FileNotFoundError(f'Donate-a-Cry not prepared: {base}')
    clips = []
    for wav in sorted(base.rglob('*.wav')):
        clips.append(Clip(path=str(wav), dataset='donateacry',
                          category=wav.parent.name, group='donateacry/infant_cry',
                          split=_hash_split(wav.stem)))
    return clips


RAVDESS_EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}


def index_ravdess() -> List[Clip]:
    base = DATA_DIR / 'ravdess'
    wavs = sorted(base.rglob('*.wav'))
    if not wavs:
        raise FileNotFoundError(f'RAVDESS not prepared: {base}')
    clips = []
    for wav in wavs:
        parts = wav.stem.split('-')
        if len(parts) != 7 or parts[1] != '01':
            continue  # speech channel only
        emotion = RAVDESS_EMOTIONS[parts[2]]
        intensity = 'strong' if parts[3] == '02' else 'normal'
        clips.append(Clip(path=str(wav), dataset='ravdess', category=emotion,
                          group=f'ravdess/{emotion}', intensity=intensity,
                          speaker=parts[6], split='dev' if int(parts[6]) <= 12 else 'test'))
    return clips


CREMAD_INTENSITY = {'LO': 'low', 'MD': 'medium', 'HI': 'high', 'XX': 'unspecified'}


def index_cremad() -> List[Clip]:
    """CREMA-D test+validation splits, materialised from the HF parquet mirror."""
    out = DATA_DIR / 'cremad'
    parquets = sorted((RAW_DIR / 'cremad').glob('*.parquet'))
    if not parquets:
        raise FileNotFoundError('CREMA-D parquet files missing in data/raw/cremad')
    index_file = out / 'index.csv'
    if not index_file.exists():
        import pandas as pd
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for pq in parquets:
            df = pd.read_parquet(pq)
            for _, r in df.iterrows():
                name = Path(r['audio']['path']).name
                dst = out / name
                if not dst.exists():
                    dst.write_bytes(r['audio']['bytes'])
                rows.append({'file': name, 'emotion': r['emotion']})
        with open(index_file, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['file', 'emotion'])
            w.writeheader()
            w.writerows(rows)
        logger.info('CREMA-D: wrote %d wav files', len(rows))
    clips = []
    with open(index_file, newline='') as fh:
        for r in csv.DictReader(fh):
            parts = Path(r['file']).stem.split('_')
            intensity = CREMAD_INTENSITY.get(parts[3], 'unspecified') if len(parts) == 4 else ''
            clips.append(Clip(path=str(out / r['file']), dataset='cremad',
                              category=r['emotion'], group=f"cremad/{r['emotion']}",
                              intensity=intensity, speaker=parts[0], split=_hash_split(parts[0])))
    return clips


def index_vivae() -> List[Clip]:
    base = DATA_DIR / 'VIVAE' / 'full_set'
    wavs = sorted(base.glob('*.wav'))
    if not wavs:
        raise FileNotFoundError(f'VIVAE not prepared: {base}')
    clips = []
    for wav in wavs:
        speaker, emotion, intensity, _ = wav.stem.split('_')
        clips.append(Clip(path=str(wav), dataset='vivae', category=emotion,
                          group=f'vivae/{emotion}', intensity=intensity, speaker=speaker,
                          split='dev' if int(speaker[1:]) <= 6 else 'test'))
    return clips


def index_librispeech(limit: int = 400, seed: int = 0) -> List[Clip]:
    base = DATA_DIR / 'LibriSpeech' / 'dev-clean'
    flacs = sorted(base.rglob('*.flac'))
    if not flacs:
        raise FileNotFoundError(f'LibriSpeech not prepared: {base}')
    rng = random.Random(seed)
    rng.shuffle(flacs)
    transcripts: Dict[str, str] = {}
    for trans in base.rglob('*.trans.txt'):
        for line in trans.read_text().splitlines():
            key, _, text = line.partition(' ')
            transcripts[key] = text.strip()
    return [Clip(path=str(f), dataset='librispeech', category='read_speech',
                 group='librispeech/read_speech', speaker=f.parent.parent.name,
                 split=_hash_split(f.parent.parent.name), transcript=transcripts.get(f.stem, ''))
            for f in flacs[:limit]]


def index_fleurs_he() -> List[Clip]:
    base = DATA_DIR / 'fleurs_he'
    tsv = base / 'dev.tsv'
    if not tsv.exists():
        raise FileNotFoundError(f'FLEURS he_il not prepared: {tsv}')
    clips = []
    with open(tsv, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if len(row) < 4:
                continue
            wav = base / 'dev' / row[1]
            if wav.exists():
                clips.append(Clip(path=str(wav), dataset='fleurs_he', category='hebrew_speech',
                                  group='fleurs/he', transcript=row[2], speaker=row[0]))
    return clips


INDEXERS: Dict[str, Callable[[], List[Clip]]] = {
    'esc50': index_esc50,
    'donateacry': index_donateacry,
    'ravdess': index_ravdess,
    'cremad': index_cremad,
    'vivae': index_vivae,
    'librispeech': index_librispeech,
    'fleurs_he': index_fleurs_he,
}

_INDEX_CACHE: Dict[str, List[Clip]] = {}


def index(name: str) -> List[Clip]:
    if name not in _INDEX_CACHE:
        _INDEX_CACHE[name] = INDEXERS[name]()
    return _INDEX_CACHE[name]


# ---------------------------------------------------------------------------
# Task manifests
# ---------------------------------------------------------------------------
@dataclass
class Labelled:
    clip: Clip
    label: int          # 1 = positive for the task, 0 = negative


# ESC-50 human non-speech categories that are the hardest negatives for a cry detector
ESC50_HUMAN_HARD = {'laughing', 'coughing', 'sneezing', 'breathing', 'snoring',
                    'clapping', 'brushing_teeth', 'drinking_sipping', 'footsteps'}
# ESC-50 loud/impulsive categories that stress the violence detector
ESC50_IMPACT = {'glass_breaking', 'door_wood_knock', 'fireworks', 'can_opening',
                'clapping', 'clock_alarm', 'siren', 'car_horn', 'chainsaw'}


def build_task(task: str, neg_limit: Optional[int] = None, seed: int = 0) -> List[Labelled]:
    """Return labelled clips for ``task`` in {'cry', 'violence', 'anger', 'speech', 'asr_he'}."""
    items: List[Labelled] = []

    def pos(clips):
        items.extend(Labelled(c, 1) for c in clips)

    def neg(clips):
        items.extend(Labelled(c, 0) for c in clips)

    if task == 'cry':
        esc = index('esc50')
        pos([c for c in esc if c.category == 'crying_baby'])
        pos(index('donateacry'))
        neg([c for c in esc if c.category != 'crying_baby'])
        neg([c for c in index('ravdess') if c.category in ('neutral', 'calm')])
        neg([c for c in index('cremad') if c.category == 'neutral'])
        neg(index('librispeech'))
    elif task == 'violence':
        viv = index('vivae')
        pos([c for c in viv if c.category in ('anger', 'fear', 'pain')
             and c.intensity in ('strong', 'peak')])
        rav = index('ravdess')
        pos([c for c in rav if c.category == 'angry' and c.intensity == 'strong'])
        neg([c for c in rav if c.category in ('neutral', 'calm', 'happy') and c.intensity == 'normal'])
        neg([c for c in index('cremad') if c.category in ('neutral', 'happy')])
        neg(index('librispeech'))
        neg([c for c in index('esc50') if c.category != 'crying_baby'])
        neg([c for c in viv if c.category in ('pleasure', 'achievement') and c.intensity in ('low', 'moderate')])
    elif task == 'anger':
        rav = index('ravdess')
        cre = index('cremad')
        pos([c for c in rav if c.category == 'angry'])
        pos([c for c in cre if c.category == 'anger'])
        neg([c for c in rav if c.category in ('neutral', 'calm', 'happy')])
        neg([c for c in cre if c.category in ('neutral', 'happy')])
        neg(index('librispeech'))
    elif task == 'speech':
        pos(index('librispeech'))
        pos([c for c in index('ravdess') if c.category in ('neutral', 'calm', 'happy', 'sad')])
        pos([c for c in index('cremad') if c.category in ('neutral', 'happy', 'sad')])
        neg([c for c in index('esc50') if not c.group.endswith('human_nonspeech')])
        neg(index('donateacry'))
    elif task == 'asr_he':
        pos(index('fleurs_he'))
    elif task == 'asr_en':
        pos([c for c in index('librispeech') if c.transcript][:200])
    else:
        raise ValueError(f'unknown task {task}')

    if neg_limit is not None:
        rng = random.Random(seed)
        negs = [i for i in items if i.label == 0]
        poss = [i for i in items if i.label == 1]
        if len(negs) > neg_limit:
            # stratified by group so every negative family stays represented
            by_group: Dict[str, List[Labelled]] = {}
            for it in negs:
                by_group.setdefault(it.clip.group, []).append(it)
            share = max(1, neg_limit // len(by_group))
            kept = []
            for g, lst in by_group.items():
                rng.shuffle(lst)
                kept.extend(lst[:share])
            negs = kept
        items = poss + negs
    return items


TASKS = ['cry', 'violence', 'anger', 'speech', 'asr_he', 'asr_en']
