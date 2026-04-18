import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx=None, task="freshness"):
        """Generate Grad-CAM heatmap for the specified task head.

        Args:
            input_tensor: Preprocessed image tensor (B, C, H, W)
            class_idx: Target class index (defaults to predicted class)
            task: One of 'freshness', 'quality', 'shelf_life', 'rotation'

        Note: target_layer must be a convolutional layer that outputs a spatial
        feature map (B, C, H, W). Do NOT pass the full backbone with global_pool
        enabled — use model.backbone.blocks[-1] instead.
        """
        task_index = {"freshness": 0, "quality": 1, "shelf_life": 2, "rotation": 3}
        idx = task_index.get(task, 0)

        self.model.eval()
        output = self.model(input_tensor)
        task_output = output[idx]

        if class_idx is None:
            if task == "shelf_life":
                class_idx = 0  # regression has single output
            else:
                class_idx = torch.argmax(task_output, dim=1).item()

        self.model.zero_grad()
        if task == "shelf_life":
            loss = task_output[0, 0]
        else:
            loss = task_output[0, class_idx]
        loss.backward()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Grad-CAM requires spatial (C, H, W) feature maps
        if gradients.ndim != 3 or activations.ndim != 3:
            raise ValueError(
                "Grad-CAM target layer must output a spatial (C, H, W) feature map. "
                "Use model.backbone.blocks[-1] instead of model.backbone."
            )

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        return heatmap

    def overlay_heatmap(
        self, heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET
    ):
        # Resize heatmap to match original image
        heatmap = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        overlayed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return overlayed_img

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
