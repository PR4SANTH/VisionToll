"""
VisionToll: Intelligent Vehicle Detection and Classification
Image-Based Automated Toll Plaza Monitoring System
Academic Context: 23CSE473 Neural Networks and Deep Learning (Group A12)

Architecture: Ultralytics YOLOv8s (Experiment 3 Champion Checkpoint)
Model Status: FROZEN FOR INFERENCE ONLY
Verified Checkpoint SHA-256: 5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD
Workflow: Static Image Upload -> YOLO Vehicle Detection -> Classification -> Census Audit
"""

import sys
import io
import time
import hashlib
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="VisionToll - Intelligent Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic/CV Dashboard Aesthetics
st.markdown("""
<style>
    /* Global Typography & Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.25);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #10b981;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10b981;
    }
    
    /* Benchmark Callout */
    .benchmark-badge {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #93c5fd;
        margin-top: 12px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Constants & Paths
ROOT_DIR = Path("D:/VisionToll")
FROZEN_MODEL_PATH = ROOT_DIR / "models" / "exp3_yolov8s" / "best.pt"
EXPECTED_SHA256 = "5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD"
DEMO_SAMPLES_DIR = ROOT_DIR / "results" / "final_project" / "demo_samples"

# 5-Class Target Taxonomy
CLASS_NAMES = {
    0: "Bus",
    1: "Car",
    2: "Motorcycle",
    3: "Auto Rickshaw",
    4: "Truck"
}

CLASS_COLORS = {
    0: "#3b82f6",  # Bus (Blue)
    1: "#10b981",  # Car (Green)
    2: "#ef4444",  # Motorcycle (Red)
    3: "#8b5cf6",  # Auto Rickshaw (Purple)
    4: "#f59e0b"   # Truck (Amber)
}

CLASS_ICONS = {
    0: "🚌",
    1: "🚗",
    2: "🏍️",
    3: "🛺",
    4: "🚚"
}

EXCLUDED_CLASSES = ["Van", "Pickup", "Scooter", "Mini-bus"]

@st.cache_resource(show_spinner=False)
def load_frozen_model():
    """Loads and verifies the frozen YOLOv8s checkpoint."""
    if not FROZEN_MODEL_PATH.exists():
        return None, "File Not Found", False
    
    try:
        with open(FROZEN_MODEL_PATH, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest().upper()
        
        is_valid = (file_hash == EXPECTED_SHA256)
        model = YOLO(str(FROZEN_MODEL_PATH))
        return model, file_hash, is_valid
    except Exception as e:
        return None, str(e), False

def main():
    # Load and Verify Model
    model, actual_hash, is_hash_valid = load_frozen_model()

    # Top Header & Status Indicator
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1 style='margin-bottom: 2px;'>VisionToll</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0; color: #94a3b8; font-weight: 400;'>Intelligent Vehicle Detection</h3>", unsafe_allow_html=True)
        st.markdown(
            "**Image-Based Vehicle Detection and Classification for Automated Toll Plaza Monitoring**  \n"
            "<span style='color: #64748b; font-size: 0.85rem;'>Academic Project: 23CSE473 Neural Networks and Deep Learning — Group A12</span>",
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        if model is not None and is_hash_valid:
            st.markdown(
                """
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>MODEL READY • YOLOv8s • Experiment 3</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif model is not None:
            st.warning("⚠️ Checkpoint hash mismatch")
        else:
            st.error("❌ Model checkpoint missing")

    st.markdown("<hr style='margin-top: 16px; margin-bottom: 20px; border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

    # Sidebar: Model Specs & Inference Controls
    with st.sidebar:
        st.markdown("### 📋 Model Information")
        st.markdown(
            """
            | Property | Specification |
            | :--- | :--- |
            | **MODEL** | YOLOv8s (Small) |
            | **SOURCE** | Experiment 3 |
            | **CLASSES** | 5 Closed Classes |
            | **INPUT** | JPG / JPEG / PNG |
            | **INFERENCE** | Image-only |
            | **RESOLUTION** | 640×640 (Standard) |
            | **STATUS** | Loaded & Ready |
            """
        )
        st.caption("ℹ️ The application runs inference using the permanently frozen final model checkpoint.")

        st.markdown("---")
        st.markdown("### 🎛️ Inference Display Settings")
        
        conf_thresh = st.slider(
            "Confidence threshold: {:.2f}".format(st.session_state.get("conf_slider", 0.25)),
            min_value=0.05,
            max_value=1.00,
            value=0.25,
            step=0.05,
            key="conf_slider",
            help="Post-processing display filter. Does not alter trained model weights or benchmark evaluation settings."
        )
        st.caption(f"Confidence threshold: **{conf_thresh:.2f}**")
        
        iou_thresh = st.slider(
            "NMS IoU Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Intersection-over-Union threshold for non-maximum suppression proposal merging."
        )

        st.markdown("---")
        st.markdown("### 🏷️ 5-Class Target Taxonomy")
        for cid, cname in CLASS_NAMES.items():
            color = CLASS_COLORS[cid]
            icon = CLASS_ICONS[cid]
            st.markdown(f"<span style='color:{color}; font-weight:600;'>{icon} Class {cid}: {cname}</span>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 10px; font-size: 0.8rem; color: #94a3b8;'><strong>Strict Exclusions:</strong></div>", unsafe_allow_html=True)
        st.caption(", ".join(EXCLUDED_CLASSES) + " (Deliberately excluded to maintain semantic taxonomy boundaries)")

    if model is None:
        st.error(f"Critical Error: Unable to initialize YOLOv8s from `{FROZEN_MODEL_PATH}`. Please verify checkpoint location.")
        st.stop()

    # Input Section
    st.markdown("### 1. Upload Vehicle Scene")
    
    input_source = st.radio(
        "Select Image Input Mode:",
        ["📁 Upload Local Image (JPG, JPEG, PNG)", "🖼️ Select Packaged Validation Demo Sample"],
        horizontal=True
    )

    image = None
    image_name = ""

    if input_source == "📁 Upload Local Image (JPG, JPEG, PNG)":
        uploaded_file = st.file_uploader(
            "Upload a highway or toll approach roadway image:",
            type=["jpg", "jpeg", "png"],
            help="Accepts standard static RGB images (JPG, JPEG, PNG). Single-frame detection only; video is outside project scope."
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error decoding uploaded image file: {e}. Please ensure it is a valid, uncorrupted JPG or PNG.")
    else:
        # Load available demo samples from validation split
        demo_files = list(DEMO_SAMPLES_DIR.glob("demo_*.jpg"))
        if demo_files:
            demo_titles = {
                "demo_01.jpg": "⭐ Demo 01 (Primary): 5-Class Complete Census (Bus, Car, Motorcycle, Auto, Truck)",
                "demo_02.jpg": "⭐ Demo 02 (Backup): 5-Class Toll Plaza Lane View (All 5 Target Classes)",
                "demo_03.jpg": "⭐ Demo 03: High-Density Motorcycle & Transit (4 Motorcycles + Bus + Auto)",
                "demo_04.jpg": "⭐ Demo 04: Clean 4-Class Exhibition (1:1 High Confidence >0.86)",
                "demo_05.jpg": "⭐ Demo 05: Dense Urban Multiclass Queue (4 Classes, 9 Vehicles)",
                "demo_crowded_scene_1152.jpg": "Demo: Crowded Scene 1152 (11 Vehicles)",
                "demo_difficult_scene_1181.jpg": "Demo: Difficult Multiclass Scene 1181",
                "demo_motorcycle_detection_1057.jpg": "Demo: Motorcycle Detection 1057",
                "demo_multiclass_scene_1181.jpg": "Demo: Multiclass Scene 1181",
                "demo_small_distant_vehicles_1105.jpg": "Demo: Small Distant Vehicles 1105",
            }
            # Prioritize demo_01 through demo_05 first
            def demo_sort_key(p):
                name = p.name
                if name.startswith("demo_0"):
                    return (0, name)
                return (1, name)

            sorted_demos = sorted(demo_files, key=demo_sort_key)
            demo_options = {p.name: p for p in sorted_demos}
            selected_sample_name = st.selectbox(
                "Choose a pre-packaged validation scene (Held-out from training):",
                options=list(demo_options.keys()),
                format_func=lambda x: demo_titles.get(x, x.replace("demo_", "").replace(".jpg", "").replace("_", " ").title())
            )
            image_path = demo_options[selected_sample_name]
            try:
                image = Image.open(image_path).convert("RGB")
                image_name = image_path.name
            except Exception as e:
                st.error(f"Error loading validation demo sample: {e}")
        else:
            st.warning("No pre-packaged demo samples found in `results/final_project/demo_samples/`.")

    # Execution & Display
    if image is not None:
        st.markdown("<hr style='margin: 16px 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
        
        # Measure Live Inference Time
        t_start = time.perf_counter()
        with st.spinner("Executing YOLOv8s neural forward pass..."):
            results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Parse Detections
        detections = []
        class_counts = {cname: 0 for cname in CLASS_NAMES.values()}
        conf_values = []

        for idx, box in enumerate(results.boxes, start=1):
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            cname = CLASS_NAMES.get(cid, f"Unknown ({cid})")
            
            if cname in class_counts:
                class_counts[cname] += 1
            conf_values.append(conf)

            detections.append({
                "#": idx,
                "Class": cname,
                "Confidence": f"{conf:.2%}",
                "Bounding Box": f"[{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}]",
                "Width (px)": xyxy[2] - xyxy[0],
                "Height (px)": xyxy[3] - xyxy[1],
                "_conf_num": conf
            })

        # Render Side-by-Side Images
        annotated_bgr = results.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        st.markdown("### 2. Detection & Visual Comparison")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Input**")
            st.image(image, caption=f"Source: {image_name} ({image.width}×{image.height} px)", use_container_width=True)

        with col_img2:
            st.markdown("**VisionToll Detection Output**")
            st.image(annotated_rgb, caption=f"YOLOv8s Predictions (conf ≥ {conf_thresh:.2f}, NMS IoU ≤ {iou_thresh:.2f})", use_container_width=True)

        # Download Button for Annotated Output
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=byte_im,
            file_name=f"visiontoll_annotated_{Path(image_name).stem}.png",
            mime="image/png",
            help="Download the annotated image containing exact YOLOv8s predictions."
        )

        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)

        # Automated Toll Plaza Vehicle Census (Metric Cards)
        st.markdown("### 3. Automated Toll Plaza Vehicle Census")
        
        m0, m1, m2, m3, m4, m5 = st.columns(6)
        with m0:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #94a3b8;">
                    <div class="metric-title" style="color: #94a3b8;">Total Detections</div>
                    <div class="metric-value" style="color: #f8fafc;">{len(detections)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m1:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #3b82f6;">
                    <div class="metric-title" style="color: #3b82f6;">🚌 Buses</div>
                    <div class="metric-value" style="color: #60a5fa;">{class_counts['Bus']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #10b981;">
                    <div class="metric-title" style="color: #10b981;">🚗 Cars</div>
                    <div class="metric-value" style="color: #34d399;">{class_counts['Car']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #ef4444;">
                    <div class="metric-title" style="color: #ef4444;">🏍️ Motorcycles</div>
                    <div class="metric-value" style="color: #f87171;">{class_counts['Motorcycle']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #8b5cf6;">
                    <div class="metric-title" style="color: #8b5cf6;">🛺 Auto Rickshaws</div>
                    <div class="metric-value" style="color: #a78bfa;">{class_counts['Auto Rickshaw']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid #f59e0b;">
                    <div class="metric-title" style="color: #f59e0b;">🚚 Trucks</div>
                    <div class="metric-value" style="color: #fbbf24;">{class_counts['Truck']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Live Performance & Confidence Stats
        avg_conf = np.mean(conf_values) if conf_values else 0.0
        st.markdown(
            f"""
            <div class="benchmark-badge" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span>⚡ <strong>Live Inference Latency:</strong> {inference_time_ms:.1f} ms (Pipeline execution)</span>
                <span>📈 <strong>Average Confidence:</strong> {avg_conf:.1%}</span>
                <span>📌 <strong>Reference Benchmark:</strong> 7.05 ms/image • 125.4 FPS (RTX 3050 Laptop GPU)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Detection Summary & Manifest
        if detections:
            st.markdown("### 4. Detection Manifest & Audit Table")
            
            # Sort detections by confidence descending
            detections_sorted = sorted(detections, key=lambda x: x["_conf_num"], reverse=True)
            for i, d in enumerate(detections_sorted, start=1):
                d["#"] = i
            
            df = pd.DataFrame(detections_sorted)
            st.dataframe(
                df[["#", "Class", "Confidence", "Bounding Box", "Width (px)", "Height (px)"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            # Rigorous No-Detection Guidance (Does not claim lowering slider guarantees detections)
            st.warning(
                "ℹ️ **No vehicles were detected for this image at the current inference settings.**\n\n"
                "Distant/small vehicles, unusual viewpoints, and images that differ from the training distribution "
                "may be difficult for the model.\n\n"
                "💡 *Recommendation*: Try an in-domain validation/demo image from the selector above to verify model behavior."
            )

    else:
        st.info("👆 Please upload a highway or toll approach scene image, or select a pre-packaged validation demo sample above.")

    # Expandable Information & Limitations Panels
    st.markdown("<hr style='margin-top: 36px; border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("ℹ️ About the Model", expanded=False):
            st.markdown(
                """
                - **Architecture**: Ultralytics YOLOv8s (Small Variant)
                - **Experimental Phase**: Experiment 3 (Model Capacity Scaling)
                - **Target Taxonomy**: 5 Closed Classes (Bus, Car, Motorcycle, Auto Rickshaw, Truck)
                - **Model Status**: Frozen Checkpoint (`models/exp3_yolov8s/best.pt`)
                - **Parameter Count**: 11,137,535 (11.1M parameters)
                - **Computational Complexity**: 28.7 GFLOPs at standard 640×640 input resolution
                - **Held-Out Test Results**:
                  - **mAP@0.50**: `0.8811` (88.11%)
                  - **mAP@0.50:0.95**: `0.7343` (73.43%)
                  - **Precision**: `0.8574` | **Recall**: `0.8017` | **F1**: `0.8286`
                  - **Motorcycle mAP@0.50:0.95**: `0.6361`
                - **SHA-256 Digest**: `5D1BE0D0F93B54CB1A7B11F71DC8CCA0FA6D883185D5361289E8772D1B57BDAD`
                """
            )
    with col_exp2:
        with st.expander("⚠️ Known Limitations", expanded=False):
            st.markdown(
                """
                - **Small/Distant Vehicles**: Subsampling strides in shallow convolutional heads (P3 stride 8) make vehicles under 32 pixels difficult to resolve.
                - **Motorcycle Variance**: Two-wheelers exhibit high silhouette variability (riders, pillions, luggage) and severe mutual occlusion.
                - **Partial Occlusion**: Highly dense vehicle queues can trigger bounding box suppression under greedy NMS.
                - **Overhead Gantry Viewpoints**: Images captured from steep overhead angles (roof-only views) represent a domain shift from the ground-level training data.
                - **Single-Frame Static Scope**: The current application performs static spatial detection only; video multi-object tracking is outside scientific scope.
                """
            )

    # Footer
    st.markdown(
        """
        <div class="footer-text">
            <strong>VisionToll</strong> — Deep Learning-Based Vehicle Detection and Classification<br>
            Academic Project — Neural Networks and Deep Learning (23CSE473 - Group A12)<br>
            Amrita Vishwa Vidyapeetham
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
