"""Build leakage-free metadata for FreshTrack.

Main set: Kaggle "fresh and stale images of fruits and vegetables" (6 produce
types x {fresh, stale}). The Kaggle release ships pre-augmented copies of each
source photo (rotated_by_N_, saltandpepper_, translation_, vertical_flip_,
"Copy of", *.jpg_N_N.jpg). Splitting per file puts copies of one photo in both
train and test, so we group copies by source photo, merge near-duplicates by
difference hash, and split by group.

External set: Kaggle "fruit and vegetable image recognition" (36 classes,
freshness unannotated). Classes that overlap the main vocabulary become a
cross-dataset produce-type test; the rest become a near-OOD set.

A stricter held-out-session protocol (data/metadata_session.json) holds out
whole capture sessions (same day / camera series) where a stratum has more
than one.

Usage:
    python -m src.data.build_splits            # everything
    python -m src.data.build_splits --session  # only rebuild the session split
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

MAIN_DIR = Path("data/downloads/fresh-and-stale-images-of-fruits-and-vegetables")
EXT_DIR = Path("data/downloads/fruit-and-vegetable-image-recognition")
OUT_MAIN = Path("data/metadata_v2.json")
OUT_EXT = Path("data/metadata_external.json")
SEED = 42
HASH_MAX_DIST = 2  # dHash bits; <=2 of 64 is a near-exact duplicate
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# External folder name -> main-set produce type
EXT_TO_MAIN = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "tomato": "tomato",
    "capsicum": "capsicum",
    "bell pepper": "capsicum",
}

_PREFIX = re.compile(
    r"^(?:copy of |rotated_by_\d+_|saltandpepper_|translation_|vertical_flip_)+"
)
_AUG_SUFFIX = re.compile(r"\.(?:jpe?g|png)_\d+_\d+\.jpg$")
_EXT = re.compile(r"\.(?:jpe?g|png|webp)$")


def source_key(filename: str) -> str:
    """Map an augmented Kaggle filename back to its source-photo name."""
    name = filename.lower()
    name = _PREFIX.sub("", name)
    name = _AUG_SUFFIX.sub("", name)
    name = _EXT.sub("", name)
    return name.strip()


_SESSION = re.compile(
    r"(screen shot \d{4}-\d\d-\d\d)|(whatsapp image \d{4}-\d\d-\d\d)|img_(\d{8})|(dsc)|(day)|([a-z\- ]+?)\s*\d"
)


def session_key(filename: str) -> str:
    """Capture session of a photo: its capture date, or its camera/name prefix.

    Different source photos from one session share background, lighting and
    often the same physical item, so they are not independent either.
    """
    k = source_key(filename)
    m = _SESSION.match(k)
    return m.group(0).strip() if m else k[:12]


def dhash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_duplicate_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 256):
    """Yield (i, j) with Hamming(hash_i, hash_j) <= max_dist, i < j."""
    h = hashes.astype(np.uint64)
    for start in range(0, len(h), chunk):
        block = h[start : start + chunk, None] ^ h[None, :]
        dist = np.unpackbits(block.view(np.uint8), axis=-1).reshape(
            block.shape[0], block.shape[1], 64
        ).sum(-1)
        ii, jj = np.nonzero(dist <= max_dist)
        for i, j in zip(ii + start, jj):
            if i < j:
                yield int(i), int(j)


def list_main():
    records = []
    for folder in sorted(p for p in MAIN_DIR.iterdir() if p.is_dir()):
        freshness, produce = folder.name.split("_", 1)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                records.append(
                    {
                        "image_path": f.as_posix(),
                        "freshness": "Fresh" if freshness == "fresh" else "Stale",
                        "produce_type": produce,
                        "source": MAIN_DIR.name,
                    }
                )
    return records


def list_external():
    records = []
    for split_dir in sorted(p for p in EXT_DIR.iterdir() if p.is_dir()):
        for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    records.append(
                        {
                            "image_path": f.as_posix(),
                            "ext_class": folder.name,
                            "produce_type": EXT_TO_MAIN.get(folder.name),
                            "source": EXT_DIR.name,
                        }
                    )
    return records


def build_groups(records):
    """Union augmented copies and near-duplicates into source-photo groups."""
    uf = UnionFind(len(records))

    by_key = {}
    for i, r in enumerate(records):
        key = (r["produce_type"], source_key(Path(r["image_path"]).name))
        by_key.setdefault(key, []).append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    # Hash-merge only within a produce type: low-texture images of different
    # produce collide at small Hamming distances and would chain into one
    # giant cross-type group.
    hashes = np.array([r["dhash"] for r in records], dtype=np.uint64)
    n_hash_pairs = 0
    for i, j in near_duplicate_pairs(hashes, HASH_MAX_DIST):
        if records[i]["produce_type"] == records[j]["produce_type"]:
            uf.union(i, j)
            n_hash_pairs += 1

    roots = [uf.find(i) for i in range(len(records))]
    root_to_gid = {r: g for g, r in enumerate(dict.fromkeys(roots))}
    return np.array([root_to_gid[r] for r in roots]), n_hash_pairs


def assign_split(groups, strata, seed):
    """Grouped, stratified ~70/15/15 split; no group spans two splits."""
    idx = np.arange(len(groups))
    test_fold = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trval, test = next(test_fold.split(idx, strata, groups))
    val_fold = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(val_fold.split(trval, strata[trval], groups[trval]))
    split = np.empty(len(groups), dtype=object)
    split[trval[tr_rel]] = "train"
    split[trval[val_rel]] = "val"
    split[test] = "test"
    return split


def split_imbalance(split, strata):
    """Sum over strata of |val share - 0.15| + |test share - 0.15|."""
    score = 0.0
    for s in np.unique(strata):
        m = strata == s
        score += abs((split[m] == "val").mean() - 0.15)
        score += abs((split[m] == "test").mean() - 0.15)
    return score


def group_split(records, n_seeds=50):
    """Pick, among n_seeds grouped splits, the one closest to 70/15/15 per stratum.

    Source groups are large (one photo session can hold hundreds of copies), so a
    single random grouped split can starve a class in val or test.
    """
    groups, n_hash_pairs = build_groups(records)
    strata = np.array([f"{r['produce_type']}|{r['freshness']}" for r in records])
    best_seed = min(
        range(SEED, SEED + n_seeds),
        key=lambda s: split_imbalance(assign_split(groups, strata, s), strata),
    )
    split = assign_split(groups, strata, best_seed)
    # Per-file split with the same proportions: reproduces the leaky protocol
    # used by the Kaggle notebooks, for the leakage ablation only.
    naive = np.random.default_rng(SEED).permutation(split)
    for i, r in enumerate(records):
        r["split"] = split[i]
        r["split_naive"] = naive[i]
        r["group_id"] = int(groups[i])
    return len(set(groups.tolist())), n_hash_pairs, best_seed


def build_session_split(src=OUT_MAIN, out=Path("data/metadata_session.json")):
    """Held-out-session protocol, derived from metadata_v2.json.

    Within each (produce type, freshness) stratum that has >=2 capture
    sessions, the session whose share is closest to 25% becomes test; strata
    with one session go entirely to train/val. Val = every 7th source group of
    the remaining images. Test therefore holds only unseen capture sessions.
    """
    meta = json.loads(src.read_text())
    images = meta["images"]
    strata = {}
    for r in images:
        r["session"] = session_key(Path(r["image_path"]).name)
        strata.setdefault((r["produce_type"], r["freshness"]), Counter())[r["session"]] += 1
    held_out = {}
    for key, sessions in strata.items():
        if len(sessions) >= 2:
            total = sum(sessions.values())
            held_out[key] = min(sessions, key=lambda s: (abs(sessions[s] / total - 0.25), s))
    for r in images:
        if held_out.get((r["produce_type"], r["freshness"])) == r["session"]:
            r["split_session"] = "test"
        else:
            r["split_session"] = "val" if r["group_id"] % 7 == 0 else "train"
    # Drop train/val images that are copies of a test image: same filename
    # source, or dHash within HASH_MAX_DIST. (Whole chained clusters are not
    # dropped: one spurious hash match can link two sessions and would remove
    # e.g. every stale-capsicum training image.)
    test = [r for r in images if r["split_session"] == "test"]
    test_keys = {(r["produce_type"], source_key(Path(r["image_path"]).name)) for r in test}
    test_h = np.array([int(r["dhash"], 16) for r in test], dtype=np.uint64)
    rest = [r for r in images if r["split_session"] != "test"]
    rest_h = np.array([int(r["dhash"], 16) for r in rest], dtype=np.uint64)
    near = np.zeros(len(rest), dtype=bool)
    for start in range(0, len(rest), 256):
        x = rest_h[start : start + 256, None] ^ test_h[None, :]
        dist = np.unpackbits(x.view(np.uint8), axis=-1).reshape(x.shape[0], x.shape[1], 64).sum(-1)
        near[start : start + 256] = (dist <= HASH_MAX_DIST).any(1)
    for r, is_near in zip(rest, near):
        if is_near or (r["produce_type"], source_key(Path(r["image_path"]).name)) in test_keys:
            r["split_session"] = "excluded"
    meta["description"] = (
        "Held-out-session split (field split_session); see build_session_split in "
        "src/data/build_splits.py."
    )
    meta["held_out_sessions"] = {f"{t}|{f}": s for (t, f), s in sorted(held_out.items())}
    out.write_text(json.dumps(meta, indent=1))
    counts = Counter(r["split_session"] for r in images)
    print(f"held-out sessions: {meta['held_out_sessions']}")
    print(f"split_session counts: {dict(counts)} -> wrote {out}")


def main():
    main_recs = list_main()
    ext_recs = list_external()
    print(f"main images: {len(main_recs)}, external images: {len(ext_recs)}")

    for r in main_recs + ext_recs:
        r["dhash"] = dhash(Path(r["image_path"]))

    n_groups, n_pairs, split_seed = group_split(main_recs)

    # Drop external images that near-duplicate any main-set image
    n_main = len(main_recs)
    combined = np.array([r["dhash"] for r in main_recs + ext_recs], dtype=np.uint64)
    contaminated = {
        j - n_main
        for i, j in near_duplicate_pairs(combined, HASH_MAX_DIST)
        if i < n_main <= j
    }
    ext_clean = [r for k, r in enumerate(ext_recs) if k not in contaminated]
    for r in ext_clean:
        r["role"] = "cross_dataset" if r["produce_type"] else "near_ood"

    for r in main_recs + ext_clean:
        r["dhash"] = f"{r['dhash']:016x}"

    types = sorted({r["produce_type"] for r in main_recs})
    OUT_MAIN.write_text(
        json.dumps(
            {
                "description": "Leakage-free grouped split of the Kaggle fresh/stale "
                "fruit and vegetable dataset (see src/data/build_splits.py).",
                "split_seed": int(split_seed),
                "produce_types": types,
                "freshness_labels": ["Fresh", "Stale"],
                "images": main_recs,
            },
            indent=1,
        )
    )
    OUT_EXT.write_text(
        json.dumps(
            {
                "description": "External evaluation images; cross_dataset = produce "
                "type in main vocabulary, near_ood = unseen produce classes.",
                "produce_types": types,
                "dropped_near_duplicates_of_main": len(contaminated),
                "images": ext_clean,
            },
            indent=1,
        )
    )

    counts = Counter((r["split"], r["freshness"]) for r in main_recs)
    print(f"source groups: {n_groups}, near-duplicate hash pairs merged: {n_pairs}, "
          f"split seed: {split_seed}")
    for s in ("train", "val", "test"):
        n = sum(v for (sp, _), v in counts.items() if sp == s)
        print(f"  {s}: {n} (fresh {counts[(s, 'Fresh')]}, stale {counts[(s, 'Stale')]})")
    print(f"external: {Counter(r['role'] for r in ext_clean)}, dropped {len(contaminated)}")
    print(f"wrote {OUT_MAIN} and {OUT_EXT}")


if __name__ == "__main__":
    import sys

    if "--session" in sys.argv:
        build_session_split()
    else:
        main()
        build_session_split()
