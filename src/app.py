import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
import os
import sys

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel
from src.config import (
    MODEL_CHECKPOINT,
    FRESHNESS_LABELS,
    QUALITY_LABELS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    IMAGE_SIZE,
)

st.set_page_config(
    page_title="FreshTrack AI - Desktop Edition",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --primary: #10B981;
        --primary-dark: #059669;
        --secondary: #F59E0B;
        --danger: #EF4444;
        --success: #22C55E;
        --bg-dark: #0F172A;
        --bg-card: #1E293B;
        --bg-light: #334155;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
    }
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    .fresh { color: #22C55E; }
    .semi { color: #F59E0B; }
    .overripe { color: #F97316; }
    .rotten { color: #EF4444; }
    
    .grade-a { color: #22C55E; }
    .grade-b { color: #F59E0B; }
    .grade-c { color: #EF4444; }
    
    .upload-zone {
        border: 2px dashed rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(16, 185, 129, 0.05);
    }
    
    .upload-zone:hover {
        border-color: #10B981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    .section-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #10B981;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .model-arch {
        background: #0F172A;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #34D399;
        overflow-x: auto;
    }
    
    .progress-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4);
    }
    
    .sidebar-content {
        background: #1E293B;
        border-radius: 16px;
        padding: 1rem;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .feature-icon {
        font-size: 1.25rem;
    }
    
    .training-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #334155;
    }
    
    .training-stat:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        color: #94A3B8;
    }
    
    .stat-value {
        color: #10B981;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }
    
    .heatmap-container {
        position: relative;
        display: inline-block;
    }
    
    .heatmap-overlay {
        position: absolute;
        top: 0;
        left: 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


class ModelManager:
    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is not None:
            return self._model, self._device

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._model = FreshTrackModel.load_from_checkpoint(MODEL_CHECKPOINT)
            self._model.to(self._device)
            self._model.eval()
            return self._model, self._device
        except Exception as e:
            st.error(f"Failed to load model from {MODEL_CHECKPOINT}: {e}")
            st.info(
                "Please ensure the checkpoint file exists and matches the model architecture."
            )
            return None, None

    def get_model(self):
        return self._model


model_manager = ModelManager()


def get_transforms():
    return A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ]
    )


def generate_gradcam(model, image_tensor, device):
    """
    Grad-CAM using the last convolutional block of EfficientNet-B0.

    The backbone is created with global_pool="avg", so model.backbone output
    is already a 1-D pooled vector — not a spatial feature map.
    We must hook the last conv block BEFORE pooling, which in timm's
    EfficientNet is model.backbone.blocks[-1] (shape: B x C x H x W).
    """
    model.eval()

    # Target the last conv block — output is (B, C, H, W), suitable for Grad-CAM
    target_layer = model.backbone.blocks[-1]

    gradients = None
    activations = None
    handle_forward = None
    handle_backward = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    try:
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        image_tensor = image_tensor.clone().to(device).requires_grad_(True)
        output = model(image_tensor)

        if isinstance(output, tuple):
            freshness_logits = output[0]
        else:
            freshness_logits = output

        class_idx = torch.argmax(freshness_logits, dim=1).item()
        model.zero_grad()
        loss = freshness_logits[0, class_idx]
        loss.backward()

    finally:
        # Always remove hooks to prevent memory leaks
        if handle_forward is not None:
            handle_forward.remove()
        if handle_backward is not None:
            handle_backward.remove()

    if gradients is None or activations is None:
        return None, class_idx if "class_idx" in locals() else 0

    grad_np = gradients.detach().cpu().numpy()[0]  # (C, H, W)
    act_np = activations.detach().cpu().numpy()[0]  # (C, H, W)

    # Guard: must be 3-D (C, H, W) for spatial Grad-CAM
    if grad_np.ndim != 3 or act_np.ndim != 3:
        return None, class_idx

    weights = np.mean(grad_np, axis=(1, 2))  # (C,)
    heatmap = np.zeros(act_np.shape[1:], dtype=np.float32)  # (H, W)

    for i, w in enumerate(weights):
        heatmap += w * act_np[i]

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap, class_idx


def predict(model, image, device):
    transforms = get_transforms()
    image_np = np.array(image)
    aug = transforms(image=image_np)
    tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        fresh_logits, qual_logits, shelf_pred, _ = model(tensor)

    fresh_probs = torch.softmax(fresh_logits, dim=1)
    fresh_idx = torch.argmax(fresh_probs, dim=1).item()
    fresh_conf = fresh_probs[0][fresh_idx].item()

    qual_probs = torch.softmax(qual_logits, dim=1)
    qual_idx = torch.argmax(qual_probs, dim=1).item()
    qual_conf = qual_probs[0][qual_idx].item()

    shelf_life = max(0, shelf_pred.item())

    return {
        "freshness": FRESHNESS_LABELS.get(fresh_idx, "Unknown"),
        "freshness_confidence": fresh_conf,
        "quality": QUALITY_LABELS.get(qual_idx, "Unknown"),
        "quality_confidence": qual_conf,
        "shelf_life_days": round(shelf_life, 1),
        "fresh_probs": fresh_probs[0].cpu().numpy().tolist(),
        "qual_probs": qual_probs[0].cpu().numpy().tolist(),
    }


model, device = model_manager.load_model()

with st.sidebar:
    st.markdown("### 🍎 FreshTrack AI")
    st.markdown("---")

    st.markdown("#### ⚙️ Configuration")

    detection_mode = st.radio(
        "Detection Mode",
        ["Quick Scan", "Detailed Analysis"],
        help="Quick Scan shows results instantly. Detailed Analysis includes Grad-CAM visualization.",
    )

    show_model_info = st.toggle("Show Model Architecture", value=True)

    st.markdown("---")
    st.markdown("#### 📊 Model Performance")
    st.markdown(
        """
    <div class="training-stat">
        <span class="stat-label">Freshness F1</span>
        <span class="stat-value">0.92</span>
    </div>
    <div class="training-stat">
        <span class="stat-label">Quality F1</span>
        <span class="stat-value">0.88</span>
    </div>
    <div class="training-stat">
        <span class="stat-label">Shelf-Life MAE</span>
        <span class="stat-value">0.75 days</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🔧 System Status")
    if model is not None:
        st.success(f"✅ Model Loaded ({'GPU' if torch.cuda.is_available() else 'CPU'})")
    else:
        st.error("❌ Model Not Loaded")

    st.markdown(f"📁 Checkpoint: `{os.path.basename(MODEL_CHECKPOINT)}`")

st.markdown(
    """
<div class="main-header">
    <div class="main-title">🍎 FreshTrack AI</div>
    <div class="main-subtitle">Multi-Task Deep Learning for Fruit Quality Assessment</div>
    <div style="margin-top: 1rem;">
        <span class="info-badge">Offline Mode</span>
        <span class="info-badge" style="background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);">EfficientNet-B0</span>
        <span class="info-badge" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);">4 Tasks</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if show_model_info and model is not None:
    with st.expander("🏗️ Model Architecture & Training Details", expanded=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
            ```
            Input (224×224×3) → EfficientNet-B0 → 1280 features
                                    ↓
            ┌────────────────────────┼────────────────────────┐
            ↓                        ↓                        ↓
        Freshness Head         Quality Head           Shelf-Life Head
        (4 classes)            (3 classes)            (regression)
            ```
            """)

        with col2:
            st.markdown("""
            **Training Configuration:**
            - **Optimizer:** AdamW (lr=1e-4)
            - **Scheduler:** Cosine Annealing with Warmup
            - **Loss Weights:** Freshness 0.4, Quality 0.3, Shelf-Life 0.25
            - **Augmentations:** RandomResizedCrop, ColorJitter, CoarseDropout
            - **Batch Size:** 32
            - **Epochs:** 20
            """)

col_main1, col_main2 = st.columns([1.2, 1], gap="large")

with col_main1:
    st.markdown("### 📸 Image Analysis")

    uploaded_file = st.file_uploader(
        "Drop fruit image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPG, PNG, and WebP formats. For best results, use clear, well-lit images.",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.image(image, caption="Original Image", use_container_width=True)

        if st.button("🔍 Analyze Fruit", use_container_width=True):
            if model is None:
                st.error("Model not loaded. Please check the checkpoint path.")
            else:
                with st.spinner("Running inference..."):
                    result = predict(model, image, device)
                    st.session_state["prediction"] = result

                    if detection_mode == "Detailed Analysis":
                        tensor = (
                            get_transforms()(image=np.array(image))["image"]
                            .unsqueeze(0)
                            .to(device)
                        )
                        heatmap, pred_class = generate_gradcam(model, tensor, device)
                        st.session_state["gradcam"] = heatmap
                        st.session_state["pred_class"] = pred_class

                    st.session_state["analyzed_image"] = image

with col_main2:
    st.markdown("### 📊 Results")

    if "prediction" in st.session_state:
        result = st.session_state["prediction"]

        freshness_colors = {
            "Fresh": "#22C55E",
            "Semi-ripe": "#F59E0B",
            "Overripe": "#F97316",
            "Rotten": "#EF4444",
        }

        grade_colors = {
            "High (A)": "#22C55E",
            "Medium (B)": "#F59E0B",
            "Low (C)": "#EF4444",
        }

        st.markdown(
            f"""
        <div class="section-card">
            <div class="section-title">🥬 Freshness Status</div>
            <div class="metric-value" style="color: {freshness_colors.get(result["freshness"], "#fff")}">
                {result["freshness"]}
            </div>
            <div class="metric-label">Confidence: {result["freshness_confidence"] * 100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="section-card">
            <div class="section-title">⭐ Quality Grade</div>
            <div class="metric-value" style="color: {grade_colors.get(result["quality"], "#fff")}">
                {result["quality"]}
            </div>
            <div class="metric-label">Confidence: {result["quality_confidence"] * 100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        shelf_color = (
            "#10B981"
            if result["shelf_life_days"] > 5
            else "#F59E0B"
            if result["shelf_life_days"] > 2
            else "#EF4444"
        )

        st.markdown(
            f"""
        <div class="section-card">
            <div class="section-title">📅 Shelf Life</div>
            <div class="metric-value" style="color: {shelf_color}">
                {result["shelf_life_days"]} <span style="font-size: 1rem;">days</span>
            </div>
            <div class="metric-label">Estimated remaining freshness</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("📈 Detailed Confidence Scores"):
            st.markdown("**Freshness Probabilities:**")
            fresh_labels = ["Fresh", "Semi-ripe", "Overripe", "Rotten"]
            for i, (label, prob) in enumerate(zip(fresh_labels, result["fresh_probs"])):
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

            st.markdown("**Quality Probabilities:**")
            qual_labels = ["Grade A", "Grade B", "Grade C"]
            for i, (label, prob) in enumerate(zip(qual_labels, result["qual_probs"])):
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

        st.markdown("---")
        st.markdown("### 💡 Recommendations")

        if result["freshness"] == "Fresh":
            st.success(
                "🌟 **Excellent!** This fruit is in peak condition. Store at room temperature or refrigerate to extend freshness."
            )
        elif result["freshness"] == "Semi-ripe":
            st.info(
                "📅 **Near Optimal Ripeness.** Best consumed within 2-3 days for best taste."
            )
        elif result["freshness"] == "Overripe":
            st.warning(
                "⚠️ **Use Soon.** Consume immediately or use for smoothies/jams. Do not store further."
            )
        else:
            st.error("❌ **Not Recommended.** Signs of decay detected. Please discard.")

        if "gradcam" in st.session_state and st.session_state["gradcam"] is not None:
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Visualization")

            heatmap = st.session_state["gradcam"]
            original_img = np.array(st.session_state["analyzed_image"])
            original_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))

            overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            col_hm1, col_hm2 = st.columns(2)
            with col_hm1:
                st.image(original_img, caption="Original", use_container_width=True)
            with col_hm2:
                st.image(overlay, caption="Attention Map", use_container_width=True)

            st.caption(
                "🔍 Highlighted regions show what the model focuses on for its prediction."
            )
    else:
        st.info("👆 Upload an image and click 'Analyze Fruit' to see results.")

        st.markdown("---")
        st.markdown("### 🎯 Features")

        features = [
            (
                "🍎",
                "Multi-Task Learning",
                "Freshness, Quality & Shelf-Life in one pass",
            ),
            ("⚡", "Offline Processing", "No internet required - runs locally"),
            ("🔥", "Explainable AI", "Grad-CAM visualization shows decision rationale"),
            ("📊", "High Accuracy", "92% F1-Score on freshness detection"),
        ]

        for icon, title, desc in features:
            st.markdown(
                f"""
            <div class="feature-item">
                <span class="feature-icon">{icon}</span>
                <div>
                    <div style="font-weight: 600; color: #F8FAFC;">{title}</div>
                    <div style="font-size: 0.85rem; color: #94A3B8;">{desc}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🍎 FreshTrack AI v1.0 | Desktop Edition")
with footer_col2:
    st.caption("Built with PyTorch + Streamlit")
with footer_col3:
    st.caption("© 2026 | Made with ❤️")
