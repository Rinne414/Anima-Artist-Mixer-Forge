"""Rebuild the bundled Danbooru tag vocabulary.

Downloads the community-maintained tag list from a1111-sd-webui-tagcomplete
(MIT-licensed repo; the data itself is factual Danbooru tag metadata:
name, category, post count, aliases) and writes it gzipped with a
provenance header to ``anima_mixer/data/danbooru_tags.csv.gz``.

The snapshot is deliberately allowed to age: Anima's training data has a
fixed cutoff, so a fresh list is not automatically a better match. Rebuild
only when the pack targets a newer base model:

    python tools/build_tag_vocab.py            # download and rebuild
    python tools/build_tag_vocab.py --from-file danbooru.csv
"""

import argparse
import datetime
import gzip
import os
import urllib.request

SOURCE_URL = (
    "https://raw.githubusercontent.com/DominikDoom/"
    "a1111-sd-webui-tagcomplete/main/tags/danbooru.csv"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, "anima_mixer", "data", "danbooru_tags.csv.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", help="use a local danbooru.csv instead of downloading")
    args = ap.parse_args()

    if args.from_file:
        with open(args.from_file, encoding="utf-8", newline="") as fh:
            data = fh.read()
        source = args.from_file
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as fh:
            data = fh.read().decode("utf-8")
        source = SOURCE_URL

    header = (
        "# Danbooru tag metadata (name,category,count,aliases); categories: "
        "0=general 1=artist 3=copyright 4=character 5=meta\n"
        f"# source: {SOURCE_URL}\n"
        f"# snapshot: {datetime.date.today().isoformat()}\n"
        "# via a1111-sd-webui-tagcomplete (MIT); tag names and post counts "
        "are factual Danbooru metadata\n"
    )
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with gzip.open(DEST, "wt", encoding="utf-8", newline="", compresslevel=9) as fh:
        fh.write(header)
        fh.write(data)

    rows = sum(1 for line in data.splitlines() if line.strip())
    size_mb = os.path.getsize(DEST) / 1024 / 1024
    print(f"wrote {DEST}: {rows} rows from {source}, {size_mb:.2f} MB gz")


if __name__ == "__main__":
    main()
