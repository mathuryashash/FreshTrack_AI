import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import base64

# Set page config
st.set_page_config(
    page_title="FreshTrack AI - Fruit Quality Monitoring",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border-color: #1b5e20;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        flex: 1;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
    st.title("FreshTrack AI")
    st.markdown("---")
    
    st.subheader("Configuration")
    backend_url = st.text_input("Backend API URL", "http://localhost:8000/predict")
    
    st.markdown("---")
    st.subheader("Storage Environment")
    
    # Weather API Integration
    # Weather API Integration
    api_key = "db05f39489951a850486d03f67750ba4"
    
    # Predefined list of major cities
    city_options = ["", "Mumbai", "Bengaluru", "Delhi", "Kolkata", "Chennai", "Hyderabad", "Pune", "London", "New York", "Tokyo", "Dubai", "Singapore", "Sydney", "Other"]
    
    selected_city = st.selectbox("Select City for Weather", city_options, help="Auto-fetch real-time temperature and humidity")
    
    city = selected_city
    if selected_city == "Other":
        city = st.text_input("Enter City Name")
    
    if city:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            w_res = requests.get(url).json()
            if w_res.get("main"):
                st.session_state['temp'] = w_res["main"]["temp"]
                st.session_state['humidity'] = w_res["main"]["humidity"]
                st.success(f"Fetched: {city} ({st.session_state['temp']}°C, {st.session_state['humidity']}%)")
            else:
                st.error("City not found in Weather API")
        except:
            st.error("Weather API Connection Error")

    # Use session state for sliders if available, else default
    default_temp = st.session_state.get('temp', 25.0)
    default_hum = st.session_state.get('humidity', 50.0)

    temp = st.slider("Temperature (°C)", -10.0, 50.0, float(default_temp), help="Ambient temperature of the storage area")
    humidity = st.slider("Humidity (%)", 0.0, 100.0, float(default_hum), help="Relative humidity level")
    
    st.markdown("---")
    st.info("💡 **Tip:** Clear lighting and high-resolution images improve prediction accuracy.")

# Header
st.title("🍎 Fruit Freshness & Quality Analyzer")
st.markdown("Predict freshness level, quality grade, and estimated shelf-life using multi-task deep learning.")

# Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Image Upload")
    uploaded_file = st.file_uploader("Upload an image of the fruit...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Analyzed Specimen", width="stretch")
        
        analyze_btn = st.button("🚀 Run Analysis")
        
        if analyze_btn:
            with st.spinner("Processing through neural network..."):
                try:
                    # Convert image for request
                    img_byte_arr = io.BytesIO()
                    if image.mode in ("RGBA", "P"):
                        image = image.convert("RGB")
                    image.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                    data = {'temp': temp, 'humidity': humidity}
                    
                    response = requests.post(backend_url, files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.session_state['prediction_result'] = res_data
                        st.session_state['last_prediction_id'] = res_data.get('prediction_id', 'N/A')
                        st.success("Analysis complete!")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")

with col2:
    st.subheader("📊 Analysis Insights")
    
    if 'prediction_result' in st.session_state:
        res = st.session_state['prediction_result']
        
        # Dashboard style metrics
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Freshness</div>
                <div class="metric-value">{res['freshness']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Quality Grade</div>
                <div class="metric-value">{res['quality_grade']}</div>
            </div>
        </div>
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Remaining Shelf-Life</div>
                <div class="metric-value">{res['shelf_life_days']} Days</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{max([v for v in res['confidence'].values() if isinstance(v, (int, float))])*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Grad-CAM Heatmap
        if 'heatmap' in res:
            st.markdown("---")
            st.subheader("🔍 Explainability: Spoilage Detection (Grad-CAM)")
            st.info("The heatmap below highlights the regions where the model detected signs of spoilage or specific quality traits.")
            
            # Decode base64 image
            heatmap_bytes = base64.b64decode(res['heatmap'])
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
            
            st.image(heatmap_img, caption="Grad-CAM Spoilage Heatmap", width="stretch")
        
        # Detailed breakdown
        # Detailed breakdown
        with st.expander("Show detailed confidence scores", expanded=True):
            st.write("### Freshness Analysis")
            fresh_breakdown = res['confidence'].get('freshness_breakdown', {})
            if fresh_breakdown:
                st.bar_chart(fresh_breakdown)
            
            st.write("### Model Confidence")
            for task, score in res['confidence'].items():
                if task == 'freshness_breakdown':
                    continue
                if isinstance(score, (int, float)):
                     st.write(f"**{task.capitalize()}:** {score*100:.2f}%")
                     st.progress(score)
        
        # Advice based on prediction
        st.markdown("---")
        st.subheader("📝 Storage Advice")
        if res['freshness'] == 'Fresh':
            st.success("🌟 This fruit is in peak condition! Can be stored at room temperature or refrigerated to extend life further.")
        elif res['freshness'] == 'Semi-ripe':
            st.info("📅 nearing optimal ripeness. Best consumed within 2-3 days.")
        elif res['freshness'] == 'Overripe':
            st.warning("⚠️ Consume immediately or use for processed products (smoothies/jams). Do not store further.")
        else:
            st.error("❌ Signs of decay detected. Not recommended for consumption.")
            
    else:
        st.info("Please upload and analyze an image to view the results.")

    # Feedback Section
    if 'prediction_result' in st.session_state:
        st.markdown("---")
        st.subheader("💬 Help us improve!")
        with st.expander("Submit Feedback / Correct Prediction"):
            with st.form("feedback_form"):
                st.write("If you think the prediction was incorrect, please provide the actual values.")
                f_freshness = st.selectbox("Correct Freshness", ["", "Fresh", "Semi-ripe", "Overripe", "Rotten"])
                f_quality = st.selectbox("Correct Quality", ["", "A", "B", "C"])
                f_shelf_life = st.number_input("Correct Shelf-Life (Days)", 0.0, 30.0, value=0.0)
                f_comments = st.text_area("Comments")
                
                submit_feedback = st.form_submit_button("Submit Feedback")
                
                if submit_feedback:
                    feedback_url = backend_url.replace("/predict", "/feedback")
                    feedback_payload = {
                        "prediction_id": st.session_state.get('last_prediction_id', "N/A"),
                        "correct_freshness": f_freshness,
                        "correct_quality": f_quality,
                        "correct_shelf_life": f_shelf_life,
                        "comments": f_comments
                    }
                    try:
                        f_res = requests.post(feedback_url, data=feedback_payload)
                        if f_res.status_code == 200:
                            st.success("Thank you for your feedback! It has been logged for model retraining.")
                        else:
                            st.error("Failed to submit feedback.")
                    except:
                        st.error("Could not connect to feedback endpoint.")

# Footer
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.caption("FreshTrack AI v1.0 | Dashboard powered by Streamlit")
with footer_col2:
    st.markdown("<p style='text-align: right;'><a href='#'>Documentation</a> | <a href='#'>GitHub Repo</a></p>", unsafe_allow_html=True)
