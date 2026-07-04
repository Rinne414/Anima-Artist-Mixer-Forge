"""Bundled Danbooru tag vocabulary: the reliable "does this tag exist" check.

Live calibration (2026-07-04) showed the encoder cannot tell known artist
tags from unknown ones, so existence is checked against factual Danbooru tag
metadata bundled at ``anima_mixer/data/danbooru_tags.csv.gz`` (name,
category, post count, aliases; see tools/build_tag_vocab.py for provenance
and rebuild instructions).

Honesty note baked into the wording: the bundled list is a filtered
snapshot, so "not found" means *typo or a small/new artist below the
snapshot's threshold* — e.g. the real artist tag "uof" is absent — never
"the model does not know it". The definitive test stays the solo A/B
(AnimaArtistABVariants + AnimaArtistImpactMap).

The vocabulary loads lazily on first lookup (~140k tags + ~30k aliases,
roughly 30 MB resident) and stays cached for the process lifetime.
"""

import csv
import gzip
import logging
import os

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "danbooru_tags.csv.gz")

CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
ARTIST_CATEGORY = 1
LOW_POST_COUNT = 50

# None = not loaded yet; False = load failed (do not retry every call);
# otherwise a (tags, aliases) pair. Tests may inject a small pair here.
_VOCAB_CACHE = None


def normalize_tag(text):
    """Prompt-style artist entry -> Danbooru tag form.

    Strips the Anima ``@`` marker and backslash escapes, converts full-width
    parentheses, lowercases, and joins whitespace runs with underscores.
    """
    s = str(text or "").strip().lower()
    if s.startswith("@"):
        s = s[1:]
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("（", "(").replace("）", ")")
    s = "_".join(s.split())
    return s.strip("_")


def load_vocab():
    """Return (tags, aliases) dicts, loading the bundled file on first use.

    ``tags`` maps a tag name to ``(category, count)``; ``aliases`` maps an
    alias to its canonical tag name. Returns (None, None) when the bundled
    file is missing or unreadable (logged once).
    """
    global _VOCAB_CACHE
    if _VOCAB_CACHE is False:
        return None, None
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE
    tags, aliases = {}, {}
    try:
        with gzip.open(DATA_PATH, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.reader(line for line in fh if not line.startswith("#")):
                if len(row) < 3:
                    continue
                name = row[0]
                try:
                    category, count = int(row[1]), int(row[2])
                except ValueError:
                    continue
                tags[name] = (category, count)
                if len(row) > 3 and row[3]:
                    for alias in row[3].split(","):
                        alias = alias.strip()
                        if alias:
                            aliases.setdefault(alias, name)
    except Exception as exc:
        logger.warning(
            "[tag_vocab] bundled Danbooru tag list unavailable (%s): %s; "
            "vocabulary checks are disabled", DATA_PATH, exc,
        )
        _VOCAB_CACHE = False
        return None, None
    # A name that is both a real tag and someone's alias counts as the tag.
    for name in tags:
        aliases.pop(name, None)
    _VOCAB_CACHE = (tags, aliases)
    logger.info(
        "[tag_vocab] loaded %d tags / %d aliases from %s",
        len(tags), len(aliases), os.path.basename(DATA_PATH),
    )
    return _VOCAB_CACHE


def lookup(name):
    """Classify one artist entry against the vocabulary.

    Returns a dict with ``status`` in {"artist", "other_category", "alias",
    "not_found", "unavailable"} plus ``canonical`` / ``category`` / ``count``
    where applicable.
    """
    tags, aliases = load_vocab()
    if tags is None or aliases is None:
        return {"status": "unavailable", "canonical": None,
                "category": None, "count": None}
    tag = normalize_tag(name)
    hit = tags.get(tag)
    if hit is not None:
        status = "artist" if hit[0] == ARTIST_CATEGORY else "other_category"
        return {"status": status, "canonical": tag,
                "category": hit[0], "count": hit[1]}
    canonical = aliases.get(tag)
    if canonical is not None:
        hit = tags.get(canonical)
        if hit is not None:
            return {"status": "alias", "canonical": canonical,
                    "category": hit[0], "count": hit[1]}
    return {"status": "not_found", "canonical": None,
            "category": None, "count": None}


def describe(name):
    """One human-readable verdict line body for an artist entry."""
    res = lookup(name)
    status = res["status"]
    if status == "unavailable":
        return "tag list unavailable"
    if status == "artist":
        note = ", low post count — style may be weak" if res["count"] < LOW_POST_COUNT else ""
        return f"known artist tag ({res['count']} posts{note})"
    if status == "alias":
        return (
            f"alias of '{res['canonical']}' ({res['count']} posts) — "
            "prefer the canonical name; aliases may encode differently"
        )
    if status == "other_category":
        kind = CATEGORY_NAMES.get(res["category"], "non-artist")
        return (
            f"a {kind} tag, not an artist — it still encodes, "
            "but it is not an artist style"
        )
    return (
        "not in the bundled Danbooru list (typo? or a small/new artist "
        "below the snapshot's threshold) — confirm with a solo A/B"
    )


def report_lines(names):
    """Vocabulary section shared by ChainPreview and TagCheck."""
    tags, _ = load_vocab()
    if tags is None:
        return [
            "danbooru tag check: unavailable (bundled tag list missing or unreadable)",
        ]
    lines = ["danbooru tag check (bundled snapshot; absence != unknown to the model):"]
    for name in names:
        lines.append(f"  {name} — {describe(name)}")
    return lines
