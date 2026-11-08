"""Fetch the UrbanSound8K ``children_playing`` slices (~0.5 GB) into data/urbansound8k/.

Only that class is needed (real kids at play: the main false-alarm source in a
kindergarten), so instead of the full 7 GB archive the rows are pulled one by one
through the Hugging Face dataset viewer of ``danavery/urbansound8K``.

    .venv/bin/python -m evaluation.fetch_urbansound_children

Licence: UrbanSound8K is CC BY-NC (research use; the audio stays in the git-ignored data/).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'urbansound8k' / 'children_playing'
VIEWER = 'https://datasets-server.huggingface.co/filter?'


def _page(offset: int) -> dict:
    q = urllib.parse.urlencode({'dataset': 'danavery/urbansound8K', 'config': 'default', 'split': 'train',
                                'where': "\"class\"='children_playing'", 'offset': offset, 'length': 100})
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(VIEWER + q, timeout=90))
        except Exception as e:                       # the viewer returns transient 500s
            print(f'page {offset}: {e}; retrying', flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'dataset viewer kept failing at offset {offset}')


def _download(row: dict) -> int:
    dst = OUT / row['slice_file_name']
    if dst.exists() and dst.stat().st_size > 1000:
        return 0
    for _ in range(4):
        try:
            data = urllib.request.urlopen(row['audio'][0]['src'], timeout=90).read()
            dst.write_bytes(data)
            return len(data)
        except Exception:
            time.sleep(3)
    print('failed', row['slice_file_name'], flush=True)
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: dict = {}
    offset, total = 0, None
    while total is None or offset < total:
        d = _page(offset)
        total = d.get('num_rows_total', 0)
        page = d.get('rows', [])
        if not page:
            break
        for r in page:
            rows[r['row']['slice_file_name']] = r['row']
        offset += len(page)
        print(f'rows {len(rows)} / {total}', flush=True)
    with ThreadPoolExecutor(8) as ex:
        nbytes = sum(ex.map(_download, rows.values()))
    meta = [{k: r[k] for k in ('slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'class')}
            for r in rows.values()]
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=0))
    print(f'{len(meta)} clips, {nbytes / 1e6:.0f} MB downloaded -> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
