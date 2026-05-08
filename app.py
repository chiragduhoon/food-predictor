import streamlit as st
from ultralytics import YOLO
import PIL.Image
import cv2
import os
from pathlib import Path

# --- UI CONFIG ---
st.set_page_config(page_title="NutriScan AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)
st.title("🍛 NutriScan: Indian Food Intelligence")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("NUTRISCAN_MODEL", ROOT_DIR / "best.pt"))

# Calorie values (numeric only for math)
CAL_VALS = {
    "Bhatura": 300, "BhindiMasala": 150, "Biryani": 350, "Chole": 250,
    "ShahiPaneer": 380, "chicken": 250, "dal": 150, "dhokla": 60,
    "gulab_jamun": 150, "idli": 50, "jalebi": 150, "modak": 150,
    "palak_paneer": 280, "poha": 250, "rice": 130, "roti": 85, "samosa": 150
}

if MODEL_PATH.exists():
    @st.cache_resource
    def load_model():
        return YOLO(str(MODEL_PATH))
    
    model = load_model()
    
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Drop food image here", type=["jpg", "jpeg", "png"])
        conf_thresh = st.slider("Sensitivity (Confidence)", 0.1, 1.0, 0.25)
        # Added IOU slider for manual control over overlapping boxes
        iou_thresh = st.slider("Overlap Filter (IOU)", 0.0, 1.0, 0.45)
        st.info("Lower IOU helps merge overlapping boxes (like Idlis close together).")

    if uploaded_file:
        img = PIL.Image.open(uploaded_file)
        
        # --- FIX 1: ADD IOU PARAMETER FOR NMS ---
        results = model.predict(source=img, conf=conf_thresh, iou=iou_thresh)
        
        # Color Fix & Display
        res_plotted = results[0].plot() 
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB) 
        
        col_main, col_stats = st.columns([2, 1])

        with col_main:
            st.image(res_rgb, caption="AI Vision Analysis", use_container_width=True)
        
        with col_stats:
            st.subheader("📊 Session Summary")
            
            # --- FIX 2: ENSURE CLEAN MATH RESET ---
            total_calories = 0
            detection_counts = {}

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # In YOLOv8/v11, results[0].boxes are already filtered by conf and iou
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    
                    detection_counts[label] = detection_counts.get(label, 0) + 1
                    total_calories += CAL_VALS.get(label, 0)

                # Show breakdown correctly
                for item, count in detection_counts.items():
                    unit_cal = CAL_VALS.get(item, 0)
                    st.write(f"**{count}x** {item} — `{count * unit_cal} kcal`")
                
                st.divider()
                st.metric("Total Estimated Calories", f"{total_calories} kcal")        
                
                if total_calories > 800:
                    st.warning("High calorie meal detected. Consider portion control!")
                else:
                    st.success("Balanced meal detected.")
            else:
                st.warning("No items detected in frame.")
else:
    st.error(
        f"Weights not found at `{MODEL_PATH}`. "
        "Place your trained model at `best.pt` or set `NUTRISCAN_MODEL`."
    )
