import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import NUM_FRESHNESS_CLASSES, NUM_PRODUCE_TYPES
from src.data.build_splits import source_key
from src.models.freshtrack_model import FreshTrackModel


class TestFreshTrackModel:
    @pytest.fixture
    def model(self):
        return FreshTrackModel(pretrained=False)

    @pytest.fixture
    def sample_batch(self):
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = {
            "freshness": torch.randint(0, NUM_FRESHNESS_CLASSES, (batch_size,)),
            "produce_type": torch.randint(0, NUM_PRODUCE_TYPES, (batch_size,)),
        }
        return images, labels

    def test_forward_returns_both_heads(self, model):
        out = model(torch.randn(2, 3, 224, 224))
        assert set(out) == {"freshness", "produce_type"}
        assert out["freshness"].shape == (2, NUM_FRESHNESS_CLASSES)
        assert out["produce_type"].shape == (2, NUM_PRODUCE_TYPES)

    def test_single_task_model_has_one_head(self):
        model = FreshTrackModel(tasks=("freshness",), pretrained=False)
        assert set(model(torch.randn(1, 3, 224, 224))) == {"freshness"}
        assert model.loss_weights == {"freshness": 1.0}

    def test_training_step_returns_scalar_loss(self, model, sample_batch):
        loss = model.training_step(sample_batch, 0)
        assert loss.dim() == 0 and loss.item() >= 0

    def test_loss_weights_sum_to_one(self, model):
        assert abs(sum(model.loss_weights.values()) - 1.0) < 1e-6

    def test_mobilenet_backbone(self):
        model = FreshTrackModel(backbone="mobilenetv3_large_100", pretrained=False)
        assert model(torch.randn(1, 3, 224, 224))["produce_type"].shape == (1, NUM_PRODUCE_TYPES)


class TestSplits:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("rotated_by_15_Screen Shot 2018-06-08 at 4.59.36 PM.png", "screen shot 2018-06-08 at 4.59.36 pm"),
            ("saltandpepper_Screen Shot 2018-06-08 at 4.59.36 PM.png", "screen shot 2018-06-08 at 4.59.36 pm"),
            ("vertical_flip_Screen Shot 2018-06-08 at 4.59.36 PM.png", "screen shot 2018-06-08 at 4.59.36 pm"),
            ("IMG_20200901_181731.jpg_0_4456.jpg", "img_20200901_181731"),
            ("Copy of IMG_20200901_181731.jpg_0_12.jpg", "img_20200901_181731"),
            ("WhatsApp Image 2020-11-07 at 1.2.3 PM.jpeg_0_1.jpg", "whatsapp image 2020-11-07 at 1.2.3 pm"),
        ],
    )
    def test_source_key_strips_augmentation_markers(self, name, expected):
        assert source_key(name) == expected

    def test_no_source_group_spans_two_splits(self):
        meta = Path("data/metadata_v2.json")
        if not meta.exists():
            pytest.skip("data/metadata_v2.json not built (run python -m src.data.build_splits)")
        splits = defaultdict(set)
        for r in json.loads(meta.read_text())["images"]:
            splits[r["group_id"]].add(r["split"])
        assert all(len(s) == 1 for s in splits.values())


class TestConfig:
    def test_freshness_labels(self):
        from src.config import FRESHNESS_LABELS, FRESHNESS_TO_IDX

        assert FRESHNESS_LABELS == {0: "Fresh", 1: "Stale"}
        assert FRESHNESS_TO_IDX["Fresh"] == 0

    def test_every_produce_type_has_shelf_life_reference(self):
        from src.config import PRODUCE_TYPES, SHELF_LIFE_REFERENCE_DAYS

        assert set(PRODUCE_TYPES) == set(SHELF_LIFE_REFERENCE_DAYS)

    def test_image_settings(self):
        from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

        assert IMAGE_SIZE == 224
        assert len(NORMALIZE_MEAN) == 3
        assert len(NORMALIZE_STD) == 3
