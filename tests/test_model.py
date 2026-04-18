import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.freshtrack_model import FreshTrackModel
from src.config import NUM_FRESHNESS_CLASSES, NUM_QUALITY_CLASSES


class TestFreshTrackModel:
    @pytest.fixture
    def model(self):
        return FreshTrackModel()

    @pytest.fixture
    def sample_batch(self):
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = {
            "freshness": torch.randint(0, NUM_FRESHNESS_CLASSES, (batch_size,)),
            "quality": torch.randint(0, NUM_QUALITY_CLASSES, (batch_size,)),
            "shelf_life": torch.rand(batch_size, 1) * 10,
            "rotation": torch.randint(0, 4, (batch_size,)),
        }
        return images, labels

    def test_model_initialization(self, model):
        assert model is not None
        assert model.backbone is not None
        assert model.freshness_head is not None
        assert model.quality_head is not None
        assert model.shelf_life_head is not None

    def test_forward_pass_shape(self, model):
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x)

        assert len(output) == 4
        fresh_logits, qual_logits, shelf_pred, rot_logits = output
        assert fresh_logits.shape == (batch_size, NUM_FRESHNESS_CLASSES)
        assert qual_logits.shape == (batch_size, NUM_QUALITY_CLASSES)
        assert shelf_pred.shape == (batch_size, 1)
        assert rot_logits.shape == (batch_size, 4)

    def test_training_step(self, model, sample_batch):
        images, labels = sample_batch

        output = model.training_step((images, labels), 0)

        assert isinstance(output, torch.Tensor)
        assert output.dim() == 0  # scalar
        assert output.item() >= 0  # loss should be non-negative

    def test_validation_step(self, model, sample_batch):
        images, labels = sample_batch

        output = model.validation_step((images, labels), 0)

        assert isinstance(output, torch.Tensor)

    def test_shelf_life_positive(self, model):
        x = torch.randn(1, 3, 224, 224)
        _, _, shelf_pred, _ = model(x)

        assert torch.all(shelf_pred >= 0)

    def test_loss_weights(self, model):
        assert model.w_fresh == 0.4
        assert model.w_quality == 0.3
        assert model.w_shelf == 0.25
        assert model.w_rot == 0.05
        assert (
            abs(model.w_fresh + model.w_quality + model.w_shelf + model.w_rot - 1.0)
            < 1e-6
        )


class TestConfig:
    def test_freshness_labels(self):
        from src.config import FRESHNESS_LABELS, FRESHNESS_TO_IDX

        assert FRESHNESS_LABELS[0] == "Fresh"
        assert FRESHNESS_TO_IDX["Fresh"] == 0
        assert len(FRESHNESS_LABELS) == 4

    def test_quality_labels(self):
        from src.config import QUALITY_LABELS, QUALITY_TO_IDX

        assert QUALITY_LABELS[0] == "High (A)"
        assert QUALITY_TO_IDX["A"] == 0
        assert len(QUALITY_LABELS) == 3

    def test_image_settings(self):
        from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

        assert IMAGE_SIZE == 224
        assert len(NORMALIZE_MEAN) == 3
        assert len(NORMALIZE_STD) == 3
