import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel
from src.config import (
    MODEL_CHECKPOINT,
    FRESHNESS_LABELS,
    PRODUCE_TYPES,
    derive_quality,
    derive_shelf_life,
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


@st.cache_resource
def load_model():
    """Load once per server process; Streamlit re-runs the script on every click."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = FreshTrackModel.load_from_checkpoint(
            MODEL_CHECKPOINT, pretrained=False, weights_only=True, map_location=device
        )
        return model.to(device).eval(), device
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_CHECKPOINT}: {e}")
        st.info("Train and promote a model first (see README).")
        return None, None


def load_reported_metrics():
    """Test metrics of the promoted experiment, as written by run_experiment.py."""
    path = Path("results/summary.json")
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("mnv3_mtl")  # deployed experiment


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

        freshness_logits = output["freshness"]

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
        logits = model(tensor)

    fresh_probs = torch.softmax(logits["freshness"], dim=1)[0].cpu()
    fresh_idx = int(fresh_probs.argmax())
    type_probs = torch.softmax(logits["produce_type"], dim=1)[0].cpu()
    type_idx = int(type_probs.argmax())
    p_fresh = float(fresh_probs[0])

    return {
        "freshness": FRESHNESS_LABELS[fresh_idx],
        "freshness_confidence": float(fresh_probs[fresh_idx]),
        "produce_type": PRODUCE_TYPES[type_idx],
        "produce_type_confidence": float(type_probs[type_idx]),
        "quality": derive_quality(p_fresh),
        "shelf_life_days": derive_shelf_life(PRODUCE_TYPES[type_idx], p_fresh),
        "fresh_probs": fresh_probs.numpy().tolist(),
        "type_probs": type_probs.numpy().tolist(),
    }


model, device = load_model()
reported = load_reported_metrics()

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
    st.markdown("#### 📊 Test-set Performance")
    if reported:
        for label, key in [
            ("Freshness macro-F1", "freshness_macro_f1"),
            ("Produce-type macro-F1", "produce_type_macro_f1"),
        ]:
            m = reported[key]
            st.markdown(
                f"""<div class="training-stat"><span class="stat-label">{label}</span>
                <span class="stat-value">{m['mean']:.3f} ± {m['std']:.3f}</span></div>""",
                unsafe_allow_html=True,
            )
        st.caption("Grouped leakage-free test split, mean ± std over seeds (results/summary.json).")
    else:
        st.caption("No evaluation results yet (run src/training/run_experiment.py).")

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
        <span class="info-badge" style="background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);">MobileNetV3-L</span>
        <span class="info-badge" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);">2 Learned Tasks</span>
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
            **Backbone:** MobileNetV3-Large (pretrained on ImageNet; EfficientNet-B0 also supported)
            ```
            Input (224×224×3) → MobileNetV3-L → 1280 features
                                    ↓
                     ┌──────────────┴──────────────┐
                     ↓                             ↓
              Freshness Head               Produce-Type Head
              (Fresh / Stale)              (6 classes)
            ```
            Quality grade and shelf-life are **heuristics** derived from
            P(fresh) and the produce type; they are not learned or validated.
            """)

        with col2:
            st.markdown("""
            **Training Configuration:**
            - **Optimizer:** AdamW (lr=3e-4, wd=1e-4)
            - **Scheduler:** 1-epoch linear warmup, cosine decay
            - **Loss Weights:** Freshness 0.5, Produce type 0.5
            - **Augmentations:** RandomResizedCrop, flips, ColorJitter, CoarseDropout
            - **Batch Size:** 64, up to 10 epochs, early stopping
            - **Split:** grouped by source photo (no leakage)
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

        freshness_colors = {"Fresh": "#22C55E", "Stale": "#EF4444"}

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
            <div class="section-title">🍅 Produce Type</div>
            <div class="metric-value">{result["produce_type"].replace("_", " ").title()}</div>
            <div class="metric-label">Confidence: {result["produce_type_confidence"] * 100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="section-card">
            <div class="section-title">⭐ Quality Grade (heuristic)</div>
            <div class="metric-value" style="color: {grade_colors.get(result["quality"], "#fff")}">
                {result["quality"]}
            </div>
            <div class="metric-label">Derived from P(fresh); not a trained grader</div>
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
            <div class="section-title">📅 Shelf Life (heuristic)</div>
            <div class="metric-value" style="color: {shelf_color}">
                ~{result["shelf_life_days"]} <span style="font-size: 1rem;">days</span>
            </div>
            <div class="metric-label">Reference days × P(fresh); not a validated prediction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("📈 Detailed Confidence Scores"):
            st.markdown("**Freshness Probabilities:**")
            for label, prob in zip(FRESHNESS_LABELS.values(), result["fresh_probs"]):
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

            st.markdown("**Produce-Type Probabilities:**")
            for label, prob in zip(PRODUCE_TYPES, result["type_probs"]):
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

        st.markdown("---")
        st.markdown("### 💡 Recommendations")

        if result["freshness"] == "Fresh":
            st.success("🌟 **Looks fresh.** Visual assessment only; check smell and texture too.")
        else:
            st.warning("⚠️ **Looks stale.** Visual signs of ageing detected; inspect before use.")
        st.caption("FreshTrack is a visual aid, not a food-safety test.")

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
                "Freshness and produce type in one pass",
            ),
            ("⚡", "Offline Processing", "Runs locally once the model is on disk"),
            ("🔥", "Explainable AI", "Grad-CAM visualization shows decision rationale"),
            ("🧪", "Leakage-free Evaluation", "Tested on photos never seen in training"),
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
