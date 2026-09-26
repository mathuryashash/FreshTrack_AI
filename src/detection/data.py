"""Box-labelled data for the produce detector (class-agnostic: one class, "produce").

Stages (run in order; each is resumable):

    python -m src.detection.data coco        # COCO 2017 subsets (produce scenes, backgrounds, val test set)
    python -m src.detection.data autolabel   # Grounding DINO boxes: Kaggle photos + non-COCO produce in COCO scenes
    python -m src.detection.data cutouts     # SAM masks inside those boxes -> RGBA single-item cutouts
    python -m src.detection.data composite   # synthetic cluttered scenes: cutouts pasted on COCO backgrounds
    python -m src.detection.data build       # data/detection/{train,val}.json
    python -m src.detection.data clfcrops    # classifier crops -> data/metadata_deploy_v210.json

COCO has boxes for five produce categories (banana, apple, orange, broccoli,
carrot). Train-2017 produce scenes join training; val-2017 scenes containing a
supported type (banana, apple, orange) are the real-world cluttered test set and
are never trained on. COCO has no tomato / pepper / bitter gourd category, so
those appear unlabelled in COCO scenes.

Output record: {"image_path", "width", "height", "boxes": [[x1, y1, x2, y2], ...],
"ignore": [[x1, y1, x2, y2], ...]}, pixel coordinates; "ignore" = crowd regions.
"""

import json
import sys
import urllib.request
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("data/detection")
COCO = Path("data/downloads/coco")
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_PRODUCE = {52: "banana", 53: "apple", 55: "orange", 56: "broccoli", 57: "carrot"}
COCO_SUPPORTED = {52, 53, 55}
COCO_FOOD = set(range(52, 62))  # banana ... cake: never use these scenes as empty backgrounds
N_BACKGROUNDS = 3000


def _download(url, dest, tries=4):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return True
        except OSError:
            continue
    return False


def _coco_records(ann, split, keep_ids):
    by_img = {}
    for a in ann["annotations"]:
        if a["image_id"] in keep_ids and a["category_id"] in COCO_PRODUCE:
            by_img.setdefault(a["image_id"], []).append(a)
    recs = []
    for im in ann["images"]:
        if im["id"] not in keep_ids:
            continue
        boxes, labels, ignore = [], [], []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if a["iscrowd"]:
                ignore.append([x, y, x + w, y + h])
            else:
                boxes.append([x, y, x + w, y + h])
                labels.append(COCO_PRODUCE[a["category_id"]])
        recs.append({"image_path": str(COCO / split / im["file_name"]), "width": im["width"],
                     "height": im["height"], "boxes": boxes, "labels": labels, "ignore": ignore,
                     "coco_id": im["id"],
                     "url": f"http://images.cocodataset.org/{split}/{im['file_name']}"})
    return recs


def stage_coco():
    zpath = COCO / "annotations_trainval2017.zip"
    if not _download(COCO_ANN_URL, zpath):
        sys.exit("could not download COCO annotations")
    with zipfile.ZipFile(zpath) as z:
        train = json.loads(z.read("annotations/instances_train2017.json"))
        val = json.loads(z.read("annotations/instances_val2017.json"))

    def cats_per_image(ann):
        c = {}
        for a in ann["annotations"]:
            c.setdefault(a["image_id"], set()).add(a["category_id"])
        return c

    tc, vc = cats_per_image(train), cats_per_image(val)
    produce_train = {i for i, c in tc.items() if c & set(COCO_PRODUCE)}
    rng = np.random.default_rng(0)
    bg_pool = sorted(i for i, c in tc.items() if not c & COCO_FOOD)
    backgrounds = set(rng.choice(bg_pool, N_BACKGROUNDS, replace=False).tolist())
    test_val = {i for i, c in vc.items() if c & COCO_SUPPORTED}

    sets = {
        "coco_train": _coco_records(train, "train2017", produce_train),
        "coco_bg": _coco_records(train, "train2017", backgrounds),
        "coco_val_test": _coco_records(val, "val2017", test_val),
    }
    todo = [(r["url"], Path(r["image_path"])) for recs in sets.values() for r in recs]
    print(f"downloading {len(todo)} COCO images", flush=True)
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda t: _download(*t), todo))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, recs in sets.items():
        recs = [r for r in recs if Path(r["image_path"]).exists()]
        (OUT / f"{name}.json").write_text(json.dumps({"images": recs}))
        n_boxes = sum(len(r["boxes"]) for r in recs)
        print(f"{name}: {len(recs)} images, {n_boxes} produce boxes", flush=True)
    print(f"failed downloads: {ok.count(False)}")


# ── Auto-labelling (Grounding DINO, Apache-2.0) ─────────────────────────────

GDINO = "IDEA-Research/grounding-dino-tiny"
SAM = "facebook/sam-vit-base"
PROMPT = {"apple": "apple", "banana": "banana", "bitter_gourd": "bitter gourd", "capsicum": "bell pepper",
          "orange": "orange", "tomato": "tomato"}
# Produce COCO has no category for; found in COCO scenes they are added as boxes.
COCO_EXTRA_PROMPT = "tomato . bell pepper . bitter gourd . lemon . cucumber . potato . onion ."
BOX_THRESHOLD, TEXT_THRESHOLD, COCO_EXTRA_MIN_SCORE = 0.35, 0.25, 0.45
SAMPLE = {"fv_fresh_rotten": 2000, "vegetable_images": 1500, "ood": 1500, "coco": 3000}


def _iou_ioa(a, b):
    """IoU and intersection-over-area-of-a for boxes a [N,4], b [M,4]."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)[:, None]
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)[None, :]
    return inter / (area_a + area_b - inter + 1e-9), inter / (area_a + 1e-9)


def _drop_group_boxes(boxes):
    """Grounding DINO often adds one box around a pile plus the items in it: drop
    any box that contains two or more other boxes."""
    if len(boxes) < 3:
        return list(range(len(boxes)))
    _, ioa = _iou_ioa(boxes, boxes)  # ioa[i, j] = share of box i inside box j
    return [j for j in range(len(boxes)) if sum(ioa[i, j] > 0.9 for i in range(len(boxes)) if i != j) < 2]


def _autolabel_jobs():
    rng = np.random.default_rng(0)
    deploy = json.loads(Path("data/metadata_deploy.json").read_text())["images"]
    ood = json.loads(Path("data/metadata_deploy_ood.json").read_text())["images"]
    jobs = []
    # one original per capture group of the plain-background Kaggle set
    seen = set()
    for r in sorted(deploy, key=lambda r: (len(Path(r["image_path"]).name), r["image_path"])):
        if r["source"] == "fresh_stale_kaggle" and r["split"] != "test" and r["group_id"] not in seen:
            seen.add(r["group_id"])
            jobs.append((r, PROMPT[r["produce_type"]] + " ."))
    # Seeded permutation prefixes: a smaller sample is a subset of a larger one.
    for src in ("fv_fresh_rotten", "vegetable_images"):
        pool = [r for r in deploy if r["source"] == src and r["split"] != "test"]
        for i in rng.permutation(len(pool))[:SAMPLE[src]]:
            jobs.append((pool[i], PROMPT[pool[i]["produce_type"]] + " ."))
    ood = [r for r in ood if r["split"] != "test"]  # the classifier's held-out OOD test set
    for i in rng.permutation(len(ood))[:SAMPLE["ood"]]:
        r = ood[i]
        jobs.append((r, r["ood_class"].replace("_", " ") + " ."))
    for r in coco_scenes():
        jobs.append((r, COCO_EXTRA_PROMPT))
    return jobs


def coco_scenes():
    """The COCO train scenes used for training (they also get Grounding DINO boxes)."""
    path = OUT / "coco_train.json"
    if not path.exists():
        return []
    recs = json.loads(path.read_text())["images"]
    return [recs[i] for i in np.random.default_rng(1).permutation(len(recs))[:SAMPLE["coco"]]]


def stage_autolabel(batch=1):  # larger batches overflow 8 GB and spill to shared memory (10x slower)
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(GDINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(device).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "autolabel.jsonl"
    done = {json.loads(l)["image_path"] for l in out_path.open()} if out_path.exists() else set()
    jobs = [j for j in _autolabel_jobs() if j[0]["image_path"] not in done]
    print(f"{len(done)} done, {len(jobs)} to label", flush=True)

    def load(r):
        # Boxes come back normalised, so a <=800 px input is rescaled to the original via
        # target_sizes; draft() lets JPEG decode straight to about that size.
        im = Image.open(r["image_path"])
        size = im.size
        im.draft("RGB", (800, 800))
        im = im.convert("RGB")
        im.thumbnail((800, 800))
        return size, im

    def prefetched(chunks, pool, ahead=16):  # bounded read-ahead; pool.map would load everything at once
        queue = deque()
        for c in chunks:
            queue.append((c, pool.submit(lambda c=c: [load(r) for r, _ in c])))
            if len(queue) > ahead:
                c0, fut = queue.popleft()
                yield c0, fut.result()
        while queue:
            c0, fut = queue.popleft()
            yield c0, fut.result()

    with out_path.open("a") as f, ThreadPoolExecutor(4) as pool:
        chunks = [jobs[s:s + batch] for s in range(0, len(jobs), batch)]
        for s, (chunk, ims) in enumerate(prefetched(chunks, pool)):
            sizes, small = zip(*ims)
            inputs = proc(images=list(small), text=[p for _, p in chunk], return_tensors="pt",
                          padding=True).to(device)
            with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                outputs = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                target_sizes=[(h, w) for w, h in sizes])
            for (r, prompt), (w, h), det in zip(chunk, sizes, res):
                boxes = det["boxes"].float().cpu().numpy().round(1).tolist()
                keep = _drop_group_boxes(boxes)
                f.write(json.dumps({"image_path": r["image_path"], "width": w, "height": h,
                                    "prompt": prompt, "boxes": [boxes[k] for k in keep],
                                    "scores": [round(float(det["scores"][k]), 3) for k in keep],
                                    "source": r.get("source", "coco"), "split": r.get("split", "train"),
                                    "produce_type": r.get("produce_type") or r.get("ood_class")}) + "\n")
            if s % 200 == 0:
                f.flush()
                print(f"{(s + 1) * batch}/{len(jobs)}", flush=True)


# ── Cutouts (SAM, Apache-2.0) and synthetic scenes ───────────────────────────

CUTOUTS = OUT / "cutouts"
COMPOSITES = OUT / "composites"
N_COMPOSITE = {"train": 12000, "val": 1000}


def _load_autolabels():
    path = OUT / "autolabel.jsonl"
    return [json.loads(l) for l in path.open()] if path.exists() else []


def _det_split(r):
    """Detector train/val for a Kaggle record, or None to leave it out. Supported
    produce keeps its classifier split (its test split is never labelled).
    Unsupported produce has only 'val'/'test' in metadata_deploy_ood.json: 'val'
    trains the detector (it must find produce the classifier cannot name, so the
    gate can reject it); 'test' is the classifier's held-out OOD test set and is
    never used."""
    if r["produce_type"] in PROMPT:
        return r["split"]
    return "train" if r["split"] == "val" else None


def stage_cutouts(batch=1):  # SAM ViT-B global attention at 1024 px: ~0.8 GB per image
    import cv2
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SamProcessor.from_pretrained(SAM)
    model = SamModel.from_pretrained(SAM).to(device).eval()
    # Single-item Kaggle photos only: the best-scoring box is the item.
    jobs = [r for r in _load_autolabels() if r["source"] != "coco" and r["boxes"] and _det_split(r)]
    index = []
    CUTOUTS.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        ims = [Image.open(r["image_path"]).convert("RGB") for r in chunk]
        boxes = [[r["boxes"][int(np.argmax(r["scores"]))]] for r in chunk]
        inputs = proc(ims, input_boxes=boxes, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            out = model(**inputs, multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu(),
                                                        inputs["reshaped_input_sizes"].cpu())
        for k, (r, im) in enumerate(zip(chunk, ims)):
            best = int(out.iou_scores[k, 0].argmax())
            if float(out.iou_scores[k, 0, best]) < 0.85:
                continue
            m = masks[k][0, best].numpy()
            x1, y1, x2, y2 = map(int, boxes[k][0])
            ys, xs = np.nonzero(m)
            if len(xs) < 400 or m[y1:y2, x1:x2].sum() < 0.35 * max((x2 - x1) * (y2 - y1), 1):
                continue
            bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            rgba = np.dstack([np.asarray(im)[by1:by2, bx1:bx2], (m[by1:by2, bx1:bx2] * 255).astype(np.uint8)])
            f = 400 / max(rgba.shape[:2])
            if f < 1:
                rgba = cv2.resize(rgba, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            name = f"{len(index):05d}.png"
            cv2.imwrite(str(CUTOUTS / name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            index.append({"file": name, "source": r["source"], "split": _det_split(r),
                          "produce_type": r["produce_type"], "image_path": r["image_path"]})
        if s % (batch * 100) == 0:
            print(f"{s + len(chunk)}/{len(jobs)} -> {len(index)} cutouts", flush=True)
    (CUTOUTS / "index.json").write_text(json.dumps(index))
    print(f"{len(index)} cutouts", flush=True)


def _clean_cutout(path):
    """Reject SAM masks that are not one whole item: a sliver (solidity < 0.85),
    a block of background (fills > 95% of its box), or a ring / pile (holes > 1%
    of the mask). On 24 hand-checked cutouts this rejected 6 of the 7 faulty
    ones and none of the good ones."""
    import cv2
    a = (cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., 3] > 127).astype(np.uint8)
    area = int(a.sum())
    contours, hier = cv2.findContours(a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not area or not contours:
        return False
    hull = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
    holes = sum(cv2.contourArea(contours[k]) for k in range(len(contours)) if hier[0][k][3] >= 0)
    return area / max(hull, 1) >= 0.85 and area / a.size <= 0.95 and holes / area <= 0.01


def _rotate_rgba(rgba, angle):
    import cv2
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _scene(rng, bg, cutouts):
    """Paste 1-6 cutouts on a background; later items occlude earlier ones.
    Returns the image and, for items at least 30% visible, their boxes and
    [produce_type, freshness, visible share]."""
    import cv2
    H, W = bg.shape[:2]
    canvas = bg.astype(np.float32)
    k = int(rng.choice([1, 1, 2, 2, 3, 4, 5, 6]))
    placed = []
    for _ in range(k):
        cut = cutouts[rng.integers(len(cutouts))]
        rgba = cv2.cvtColor(cv2.imread(str(CUTOUTS / cut["file"]), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        if rng.random() < 0.5:
            rgba = rgba[:, ::-1]
        rgba = _rotate_rgba(np.ascontiguousarray(rgba), rng.uniform(0, 360))
        side = rng.uniform(0.15, 0.7 if k <= 2 else 0.45) * min(H, W)
        f = side / max(rgba.shape[:2])
        rgba = cv2.resize(rgba, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        h, w = rgba.shape[:2]
        if h < 12 or w < 12 or h >= H or w >= W:
            continue
        y, x = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        alpha = cv2.GaussianBlur(rgba[..., 3].astype(np.float32) / 255, (3, 3), 0)[..., None]
        rgb = np.clip(rgba[..., :3].astype(np.float32) * rng.uniform(0.75, 1.2) + rng.uniform(-15, 15), 0, 255)
        region = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = alpha * rgb + (1 - alpha) * region
        m = np.zeros((H, W), bool)
        m[y:y + h, x:x + w] = rgba[..., 3] > 127
        for p in placed:
            p["mask"] &= ~m
        placed.append({"mask": m, "full": int(m.sum()), "type": cut["produce_type"],
                       "freshness": cut.get("freshness")})
    boxes, items = [], []
    for p in placed:
        vis = p["mask"].sum() / p["full"] if p["full"] else 0
        if vis >= 0.3:
            ys, xs = np.nonzero(p["mask"])
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
            items.append([p["type"], p["freshness"], round(float(vis), 3)])
    img = canvas
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), boxes, items


def stage_composite():
    import cv2
    rng = np.random.default_rng(0)
    cut = [c for c in json.loads((CUTOUTS / "index.json").read_text()) if _clean_cutout(CUTOUTS / c["file"])]
    print(f"{len(cut)} cutouts pass the shape filter", flush=True)
    fresh = {r["image_path"]: r.get("freshness")
             for r in json.loads(Path("data/metadata_deploy.json").read_text())["images"]}
    for c in cut:  # None for unsupported produce and for photos without a freshness label
        c["freshness"] = fresh.get(c["image_path"])
    bgs = json.loads((OUT / "coco_bg.json").read_text())["images"]
    n_val_bg = len(bgs) // 10
    pools = {"train": ([c for c in cut if c["split"] == "train"], bgs[n_val_bg:]),
             "val": ([c for c in cut if c["split"] == "val"], bgs[:n_val_bg])}
    for split, (cuts, backs) in pools.items():
        d = COMPOSITES / split
        d.mkdir(parents=True, exist_ok=True)
        recs = []
        for i in range(N_COMPOSITE[split]):
            bg = cv2.cvtColor(cv2.imread(backs[rng.integers(len(backs))]["image_path"]), cv2.COLOR_BGR2RGB)
            if rng.random() < 0.5:
                bg = bg[:, ::-1]
            f = 640 / max(bg.shape[:2])
            bg = cv2.resize(np.ascontiguousarray(bg), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            img, boxes, items = _scene(rng, bg, cuts)
            path = d / f"{i:05d}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
            recs.append({"image_path": str(path), "width": img.shape[1], "height": img.shape[0],
                         "boxes": boxes, "items": items, "ignore": [], "source": "composite"})
            if i % 1000 == 0:
                print(f"{split} {i}", flush=True)
        (OUT / f"composite_{split}.json").write_text(json.dumps({"images": recs}))
        print(f"{split}: {len(recs)} scenes from {len(cuts)} cutouts", flush=True)


def stage_build():
    """Merge sources into train/val. COCO val (the test set) is never included."""
    rng = np.random.default_rng(0)
    load = lambda n: json.loads((OUT / n).read_text())["images"]
    coco = coco_scenes()
    extra = {r["image_path"]: r for r in _load_autolabels() if r["source"] == "coco"}
    for r in coco:  # add confident Grounding DINO boxes for produce COCO has no category for
        e = extra.get(r["image_path"])
        if not e or not e["boxes"]:
            continue
        cand = [b for b, s in zip(e["boxes"], e["scores"]) if s >= COCO_EXTRA_MIN_SCORE
                and (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * r["width"] * r["height"]]
        if cand and r["boxes"]:
            iou, ioa = _iou_ioa(cand, r["boxes"])
            cand = [b for b, i, a in zip(cand, iou.max(1), ioa.max(1)) if i < 0.3 and a < 0.7]
        r["boxes"] = r["boxes"] + cand
        r["n_auto_boxes"] = len(cand)
    order = rng.permutation(len(coco))
    n_val = len(coco) // 20
    kaggle = [dict(r, ignore=[]) for r in _load_autolabels() if r["source"] != "coco" and r["boxes"]]
    bgs = [dict(r, boxes=[], ignore=[]) for r in load("coco_bg.json")]
    n_val_bg = len(bgs) // 10
    sets = {
        "train": load("composite_train.json") + [coco[i] for i in order[n_val:]]
        + [r for r in kaggle if _det_split(r) == "train"] + bgs[n_val_bg:],
        "val": load("composite_val.json") + [coco[i] for i in order[:n_val]]
        + [r for r in kaggle if _det_split(r) == "val"] + bgs[:n_val_bg],
    }
    for name, recs in sets.items():
        (OUT / f"{name}.json").write_text(json.dumps({"images": recs}))
        print(f"{name}: {len(recs)} images, {sum(len(r['boxes']) for r in recs)} boxes, "
              f"{sum(not r['boxes'] for r in recs)} empty", flush=True)


# ── Classifier crops ─────────────────────────────────────────────────────────

CLF_CROPS = Path("data/detection/clf_crops")
CLF_META = Path("data/metadata_deploy_v210.json")
MIN_SIDE = 32  # px; smaller boxes make meaningless crops
CAP = {"coco": 2000, "composite_train": 6000, "composite_val": 600}  # per type / total / total


def stage_clfcrops():
    """Crops of supported produce in cluttered scenes, cut exactly as the app cuts
    them (detector.crop_box), added to the deployment classifier's training set.
    The classifier was trained on photos where one item fills the frame; on crops
    of real scenes it named the type right only 65% of the time.

    Sources: COCO train2017 human boxes (banana, apple, orange; type only; every
    10th COCO image id -> val) and the synthetic scenes (items >= 70% visible;
    type, plus freshness when the cutout's photo has it). COCO val2017, the web
    photos and the classifier's test splits are never used."""
    import cv2
    from collections import Counter

    from src.detection.detector import crop_box

    rng = np.random.default_rng(4)
    CLF_CROPS.mkdir(parents=True, exist_ok=True)
    recs, per_type = [], Counter()

    def save(img, box, name, w, h):
        x1, y1, x2, y2 = crop_box(box, w, h)
        path = CLF_CROPS / name
        cv2.imwrite(str(path), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(path)

    coco = json.loads((OUT / "coco_train.json").read_text())["images"]
    for i in rng.permutation(len(coco)):
        r, img = coco[i], None
        for k, (b, label) in enumerate(zip(r["boxes"], r["labels"])):
            if label not in ("banana", "apple", "orange") or per_type[label] >= CAP["coco"]:
                continue
            if min(b[2] - b[0], b[3] - b[1]) < MIN_SIDE:
                continue
            img = cv2.imread(r["image_path"]) if img is None else img
            recs.append({"image_path": save(img, b, f"coco_{r['coco_id']}_{k}.jpg", r["width"], r["height"]),
                         "produce_type": label, "freshness": None,
                         "split": "val" if r["coco_id"] % 10 == 0 else "train",
                         "group_id": f"coco:{r['coco_id']}", "source": "coco_crop"})
            per_type[label] += 1
    for split in ("train", "val"):
        scenes = json.loads((OUT / f"composite_{split}.json").read_text())["images"]
        cand = [(r, k) for r in scenes for k, (t, _, vis) in enumerate(r["items"])
                if t in PROMPT and vis >= 0.7 and min(r["boxes"][k][2] - r["boxes"][k][0],
                                                      r["boxes"][k][3] - r["boxes"][k][1]) >= MIN_SIDE]
        for i in rng.permutation(len(cand))[:CAP[f"composite_{split}"]]:
            r, k = cand[i]
            t, freshness, _ = r["items"][k]
            img = cv2.imread(r["image_path"])
            name = f"comp_{split}_{Path(r['image_path']).stem}_{k}.jpg"
            recs.append({"image_path": save(img, r["boxes"][k], name, r["width"], r["height"]),
                         "produce_type": t, "freshness": freshness, "split": split,
                         "group_id": f"composite:{split}:{Path(r['image_path']).stem}", "source": "composite_crop"})
    deploy = json.loads(Path("data/metadata_deploy.json").read_text())
    CLF_META.write_text(json.dumps({
        "description": deploy["description"] + " + crops of produce in cluttered scenes "
                                               "(src/detection/data.py clfcrops)",
        "images": deploy["images"] + recs}))
    print(f"{len(recs)} crops ({Counter((r['source'], r['split']) for r in recs)}), "
          f"types {Counter(r['produce_type'] for r in recs)}", flush=True)


if __name__ == "__main__":
    {"coco": stage_coco, "autolabel": stage_autolabel, "cutouts": stage_cutouts,
     "composite": stage_composite, "build": stage_build, "clfcrops": stage_clfcrops}[sys.argv[1]]()
