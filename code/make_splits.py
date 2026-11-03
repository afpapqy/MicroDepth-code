"""Regenerate the MicroDepth session-disjoint partition from its recorded rule and seed.

Rule (recorded in splits/split_manifest.json). Categories are taken in the recorded order. Within a category the
sessions are sorted by name. One numpy.random.default_rng(seed) stream, shared by all categories, draws
rng.permutation(n_sessions). The first permuted session is the test session, the second is the validation session
if the category has at least three sessions, and the remaining sessions are training sessions. All frames of a
session stay together, so both views of every frame are in the same list.

The lists keep the order of the frame list (frame_paths.csv) and use CRLF line endings, as in splits/*.txt.
The result is compared with the SHA-256 values recorded in the manifest. The exit status is 1 if any list differs.

Usage, from the package directory:
    python code/make_splits.py --frames frame_paths.csv --manifest splits/split_manifest.json --out NEW_OUTPUT_DIR
The output directory must not exist yet."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--frames', required=True,
                    help='CSV with one release frame path (data/CATEGORY/SESSION/FRAME) per row, column "release_path"')
    ap.add_argument('--manifest', required=True, help='splits/split_manifest.json')
    ap.add_argument('--out', required=True, help='new directory for train.txt, validation.txt and test.txt')
    a = ap.parse_args()

    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    order, seed = manifest['category_order'], manifest['seed']
    with open(a.frames, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    frames = [r['release_path'].replace('\\', '/') for r in rows]

    by_session = {}
    for frame in frames:
        _, category, session, _ = frame.split('/')
        by_session.setdefault((category, session), []).append(frame)

    rng = np.random.default_rng(seed)
    session_part = {}
    for category in order:
        sessions = sorted(s for (c, s) in by_session if c == category)
        picks = list(rng.permutation(len(sessions)))
        test = sessions[picks[0]]
        validation = sessions[picks[1]] if len(sessions) >= 3 else None
        for s in sessions:
            session_part[(category, s)] = 'test' if s == test else ('validation' if s == validation else 'train')

    lists = {'train': [], 'validation': [], 'test': []}
    for frame in frames:
        _, category, session, _ = frame.split('/')
        lists[session_part[(category, session)]].append(frame)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    identical = True
    for name, items in lists.items():
        data = ''.join(x + '\r\n' for x in items).encode('utf-8')
        (out / f'{name}.txt').write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        recorded = manifest['files_sha256'].get(f'{name}.txt')
        match = digest == recorded
        identical &= match
        print(f'{name}: {len(items)} frames, sha256 {digest[:16]}, matches manifest: {match}')
    sys.exit(0 if identical else 1)


if __name__ == '__main__':
    main()
