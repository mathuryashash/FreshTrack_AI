"""Training set for the deployed app model: the grouped main split plus real-world photos.

The research model (trained on the Kaggle fresh/stale set only) rejected or
misclassified most everyday phone photos: the source set has ~35 capture
sessions of single items on plain backgrounds. The deployed model therefore also
trains on two public datasets of real-world photos:

  data/downloads/realworld/fv_fresh_rotten  (Kaggle muhriddinmuxiddinov/fruits-and-vegetables-dataset)
      Fresh/Rotten apple, banana, orange, bell pepper, tomato (+ other produce)
  data/downloads/realworld/vegetables       (Kaggle misrakahmed/vegetable-image-dataset)
      bitter gourd, capsicum, tomato (+ other vegetables); freshness not annotated

Supported produce joins training (freshness None = type-only, the loss skips it);
unsupported produce becomes a real-photo near-OOD set for gate calibration/testing.
The research experiments in results/ are unaffected.

    python -m src.data.build_deploy_set   # -> data/metadata_deploy.json, data/metadata_deploy_ood.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.build_splits import HASH_MAX_DIST, IMAGE_EXTS, UnionFind, dhash, near_duplicate_pairs

MAIN = Path("data/metadata_v2.json")
FV_DIR = Path("data/downloads/realworld/fv_fresh_rotten")
VEG_DIR = Path("data/downloads/realworld/vegetables")
OUT = Path("data/metadata_deploy.json")
OUT_OOD = Path("data/metadata_deploy_ood.json")
SEED = 0

FV_TYPES = {"apple": "apple", "banana": "banana", "orange": "orange", "tomato": "tomato",
            "bellpepper": "capsicum", "capsicum": "capsicum"}
VEG_TYPES = {"bitter_gourd": "bitter_gourd", "capsicum": "capsicum", "tomato": "tomato"}
_FV_FOLDER = re.compile(r"^(fresh|rotten)[_ ]?(.+)$", re.I)


def _images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _group_and_split(records, fractions=(0.8, 0.1, 0.1)):
    """Cluster by file stem + dHash (within a type), then split clusters per stratum."""
    uf = UnionFind(len(records))
    by_key = {}
    for i, r in enumerate(records):
        # Same stem = same photo saved twice (e.g. "freshApple (1).jpg" and ".png")
        key = (r["produce_type"], Path(r["image_path"]).stem.lower(), Path(r["image_path"]).parent.name)
        by_key.setdefault(key, []).append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    hashes = np.array([r["dhash"] for r in records], dtype=np.uint64)
    for i, j in near_duplicate_pairs(hashes, HASH_MAX_DIST):
        if records[i]["produce_type"] == records[j]["produce_type"]:
            uf.union(i, j)
    groups = [uf.find(i) for i in range(len(records))]
    rng = np.random.default_rng(SEED)
    strata = {}
    for g, r in zip(groups, records):
        strata.setdefault((r["produce_type"], r["freshness"]), set()).add(g)
    split_of = {}
    for gs in strata.values():
        gs = sorted(gs)
        rng.shuffle(gs)
        n_val = max(1, round(len(gs) * fractions[1]))
        n_test = max(1, round(len(gs) * fractions[2]))
        for k, g in enumerate(gs):
            split_of.setdefault(g, "val" if k < n_val else "test" if k < n_val + n_test else "train")
    for g, r in zip(groups, records):
        r["split"] = split_of[g]
        r["group_id"] = f"{r['source']}:{g}"


def main():
    main_imgs = json.loads(MAIN.read_text())["images"]
    records = [
        {"image_path": r["image_path"], "freshness": r["freshness"], "produce_type": r["produce_type"],
         "split": r["split"], "group_id": f"main:{r['group_id']}", "source": "fresh_stale_kaggle"}
        for r in main_imgs
    ]
    ood = []

    fv = []
    for folder in sorted(p for p in FV_DIR.rglob("*") if p.is_dir()):
        m = _FV_FOLDER.match(folder.name)
        if not m:
            continue
        fresh = "Fresh" if m.group(1).lower() == "fresh" else "Stale"
        kind = m.group(2).lower().replace(" ", "")
        for p in _images(folder):
            rec = {"image_path": p.as_posix(), "source": "fv_fresh_rotten", "dhash": dhash(p)}
            if kind in FV_TYPES:
                fv.append({**rec, "freshness": fresh, "produce_type": FV_TYPES[kind]})
            else:
                ood.append({**rec, "ood_class": kind})
    _group_and_split(fv)

    veg = []
    for split_dir in sorted(p for p in VEG_DIR.rglob("*") if p.is_dir() and p.name in ("train", "validation", "test")):
        split = {"validation": "val"}.get(split_dir.name, split_dir.name)
        for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            kind = folder.name.lower().replace(" ", "_")
            for p in _images(folder):
                rec = {"image_path": p.as_posix(), "source": "vegetable_images", "split": split}
                if kind in VEG_TYPES:
                    veg.append({**rec, "freshness": None, "produce_type": VEG_TYPES[kind],
                                "group_id": f"veg:{p.as_posix()}"})
                else:
                    ood.append({**rec, "ood_class": kind})

    # OOD images: deterministic val/test halves (fv ones have no split yet)
    rng = np.random.default_rng(SEED)
    for r in ood:
        r.setdefault("split", "val" if rng.random() < 0.5 else "test")
        r["split"] = "val" if r["split"] in ("train", "val") else "test"
        r.pop("dhash", None)
    for r in fv:
        r.pop("dhash", None)

    all_recs = records + fv + veg
    OUT.write_text(json.dumps({
        "description": "Deployment training set: grouped main split + real-world photos (src/data/build_deploy_set.py)",
        "images": all_recs}, indent=1))
    OUT_OOD.write_text(json.dumps({
        "description": "Real-photo unsupported produce for gate calibration (val) and testing (test)",
        "images": ood}, indent=1))
    c = Counter((r["source"], r["split"]) for r in all_recs)
    print("deploy set:", dict(sorted(c.items())))
    print("types:", dict(Counter(r["produce_type"] for r in all_recs if r["split"] == "train")))
    print("freshness-unlabelled train images:", sum(r["freshness"] is None and r["split"] == "train" for r in all_recs))
    print("ood:", dict(Counter(r["split"] for r in ood)), "classes:", sorted({r["ood_class"] for r in ood}))


if __name__ == "__main__":
    main()
