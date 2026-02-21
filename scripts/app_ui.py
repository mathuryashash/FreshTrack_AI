import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys
import glob
import requests

# Add project root to path
sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel

# Page Config
st.set_page_config(page_title="FreshTrack AI - Model Tester", page_icon="🍃", layout="wide")

st.title("🍃 FreshTrack AI - Model Tester")
st.write("Upload an image of a fruit to analyze its Freshness, Quality, and Shelf Life.")

# --- Weather Integration (Open-Meteo - No Key Required) ---
st.sidebar.header("🌍 Local Weather Context")
city = st.sidebar.text_input("Enter City Name", "Bengaluru")

if city:
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5)
        if geo_res.status_code == 200 and 'results' in geo_res.json():
            location = geo_res.json()['results'][0]
            lat, lon = location['latitude'], location['longitude']
            
            # 2. Weather Data
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            w_res = requests.get(weather_url, timeout=5)
            
            if w_res.status_code == 200:
                current_weather = w_res.json()['current_weather']
                temp = current_weather['temperature']
                # Open-Meteo doesn't give humidity in 'current_weather' easily without full hourly data, 
                # but we can infer or use a different param. Let's use hourly for humidity.
                
                # Fetching humidity from hourly endpoint
                hum_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=relativehumidity_2m&current_weather=true"
                h_res = requests.get(hum_url, timeout=5)
                humidity = 50 # Default
                if h_res.status_code == 200:
                     # Get humidity for current hour (approx)
                     humidity = h_res.json()['hourly']['relativehumidity_2m'][0]

                st.sidebar.metric("Temperature", f"{temp}°C")
                st.sidebar.metric("Humidity", f"{humidity}%")
                
                if temp > 30 or humidity > 80:
                    st.sidebar.warning("⚠️ High heat/humidity may accelerate spoilage!")
            else:
                st.sidebar.error("Weather unavailable.")
        else:
            st.sidebar.error("City not found.")
    except Exception as e:
        st.sidebar.error(f"Weather Error: {e}")

# --- Model Loading ---
@st.cache_resource
def load_model():
    checkpoint_dir = os.path.join("models", "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoints:
        st.error("No checkpoints found! Please train the model first.")
        return None
        
    # Prefer robust checkpoints
    robust_checkpoints = [ckpt for ckpt in checkpoints if "stage3" not in ckpt]
    
    if robust_checkpoints:
        latest = max(robust_checkpoints, key=os.path.getmtime)
        st.sidebar.success(f"Model: {os.path.basename(latest)}")
    else:
        latest = max(checkpoints, key=os.path.getmtime)
        st.sidebar.warning(f"Model: {os.path.basename(latest)} (Stage 3)")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreshTrackModel.load_from_checkpoint(latest)
    model.to(device)
    model.eval()
    return model, device

# Load Model
model, device = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col_img, col_stats = st.columns([1, 2])
        
        image = Image.open(uploaded_file).convert("RGB")
        with col_img:
            # FIX: Replaced use_column_width=True with use_container_width=True
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Preprocess
        transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with st.spinner('Analyzing...'):
                fresh_logits, qual_logits, shelf_pred, _ = model(tensor)
                
                # --- Probabilities ---
                fresh_probs = torch.softmax(fresh_logits, dim=1)
                qual_probs = torch.softmax(qual_logits, dim=1)
                
                probs = {i: fresh_probs[0][i].item() for i in range(4)}
                # 0: Fresh, 1: Semi-ripe, 2: Overripe, 3: Rotten
                
                fresh_class_idx = torch.argmax(fresh_probs, dim=1).item()
                
                # --- Strict Freshness Adjustments ---
                # If prediction is Fresh, but hidden signal exists for spoilage:
                if fresh_class_idx == 0:
                    score = probs[2] * 1.5 + probs[3] * 2.5 # weighted badness score
                    if score > 0.15: # Threshold mechanism
                        if probs[3] > probs[2]:
                            fresh_class_idx = 3 # Rotten
                        else:
                            fresh_class_idx = 2 # Overripe
                    elif probs[1] > 0.30:
                        fresh_class_idx = 1 # Semi-ripe

                # --- Strict Quality Linking ---
                # Quality MUST correlate with Freshness.
                qual_class_idx = torch.argmax(qual_probs, dim=1).item()
                
                if fresh_class_idx == 3: # Rotten -> Force Grade C or B
                     if qual_class_idx == 0: qual_class_idx = 2 # Downgrade A to C
                elif fresh_class_idx == 2: # Overripe -> Max Grade B
                     if qual_class_idx == 0: qual_class_idx = 1 # Downgrade A to B
                
                shelf_life = shelf_pred.item()
                
                # Shelf Life Adjustments based on new class
                if fresh_class_idx == 3: shelf_life = 0.0
                elif fresh_class_idx == 2: shelf_life = min(shelf_life, 2.0)
                elif fresh_class_idx == 1: shelf_life = min(shelf_life, 5.0)
                
                freshness_map = {0: 'Fresh', 1: 'Semi-ripe', 2: 'Overripe', 3: 'Rotten'}
                quality_map = {0: 'A', 1: 'B', 2: 'C'}
                
                # Display Results
                with col_stats:
                    st.subheader("Analysis Results")
                    
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        final_label = freshness_map.get(fresh_class_idx, "Unknown")
                        st.metric("Freshness", final_label)
                        st.progress(probs[fresh_class_idx])
                        if fresh_class_idx != torch.argmax(fresh_probs, dim=1).item():
                             st.caption("⚠️ Adjusted for strictness")
                        
                    with m2:
                        st.metric("Quality Grade", quality_map.get(qual_class_idx, "Unknown"))
                        
                        # Show raw prob for the original max, OR the new forced class?
                        # Let's show the prob of the current class if valid, else 1.0 (forced)
                        q_prob = qual_probs[0][qual_class_idx].item()
                        st.progress(q_prob)
                        
                    with m3:
                        # Even Tighter Range: +/- 0.3 days
                        min_days = max(0, shelf_life - 0.3)
                        max_days = shelf_life + 0.3
                        if shelf_life <= 0.1:
                           st.metric("Shelf Life", "0 Days")
                        else:
                           st.metric("Shelf Life (Range)", f"{min_days:.1f} - {max_days:.1f} Days")

                # --- Reasoning & Safety ---
                st.divider()
                st.subheader("📝 Safety & Reasoning")
                
                with st.expander("See Detailed Probabilities"):
                     st.write("Model Confidence Breakdown:")
                     st.bar_chart({"Fresh": probs[0], "Semi-ripe": probs[1], "Overripe": probs[2], "Rotten": probs[3]})
                
                freshness_label = freshness_map.get(fresh_class_idx, "Unknown")
                quality_label = quality_map.get(qual_class_idx, "Unknown")
                
                reasoning = []
                is_safe = False
                
                # Constructed Reasoning
                if freshness_label in ['Rotten']:
                    reasoning.append("Classified as **Rotten**. Detected signs of spoilage/decay.")
                elif freshness_label in ['Overripe']:
                    is_safe = True
                    reasoning.append("Classified as **Overripe**. Fruit is aging and may be soft.")
                elif freshness_label in ['Semi-ripe']:
                    is_safe = True
                    reasoning.append("Classified as **Semi-ripe**. Developing sweetness.")
                else:
                    is_safe = True
                    reasoning.append("Classified as **Fresh**. Peak condition.")

                if quality_label == 'C':
                    reasoning.append("Quality downgraded to **Grade C** due to visible defects or shape.")
                elif quality_label == 'B':
                    reasoning.append("Quality is **Grade B** (minor imperfections).")
                else:
                    reasoning.append("**Premium Grade A** quality.")

                if is_safe:
                    st.success(f"✅ **Safe to Eat**\n\n{' '.join(reasoning)}")
                else:
                    st.error(f"❌ **Not Safe to Eat**\n\n{' '.join(reasoning)}")
