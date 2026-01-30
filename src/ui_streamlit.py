"""Modern Streamlit UI for Wound-AI Classification.

How to run:
  1. Start the FastAPI server:
     uvicorn src.api:app --reload

  2. In a separate terminal, start Streamlit:
     streamlit run src/ui_streamlit.py

  3. Open browser at http://localhost:8501
"""

from pathlib import Path
from typing import Dict, Optional

import requests
import streamlit as st
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_BATCH_PREDICT_URL = f"{API_BASE_URL}/batch_predict"

# Class color mapping (for visual coding)
CLASS_COLORS = {
    "BG": "#6B7280",  # Gray
    "D": "#EF4444",  # Red-ish
    "N": "#3B82F6",  # Blue
    "P": "#F59E0B",  # Yellow/Amber
    "S": "#8B5CF6",  # Purple
    "V": "#10B981",  # Green
}

CLASS_LABELS = {
    "BG": "Background",
    "D": "D-type wound",
    "N": "N-type wound",
    "P": "P-type wound",
    "S": "S-type wound",
    "V": "V-type wound",
}

# ============================================================================
# Custom CSS - Modern Glassmorphic Design
# ============================================================================

CUSTOM_CSS = """
<style>
/* Import Google Font - Inter */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Import Bootstrap Icons */
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

/* Global styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main content area */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
}

/* Streamlit containers */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Title styling */
h1 {
    color: white;
    font-weight: 700;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
}

h2, h3 {
    color: white;
    font-weight: 600;
    margin-top: 1.5rem;
}

/* Glassmorphic card effect */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    margin-bottom: 1.5rem;
}

/* KPI card for batch dashboard */
.kpi-card {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    margin: 0.5rem 0;
}

.kpi-label {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.9);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Prediction result card */
.result-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(15px);
    border-radius: 28px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.prediction-class {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

.prediction-prob {
    font-size: 1.5rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.95);
}

/* Probability bar container */
.prob-bar-container {
    margin: 1rem 0;
    padding: 0.75rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 16px;
}

.prob-bar-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
    color: white;
    font-weight: 500;
}

.prob-bar {
    height: 28px;
    border-radius: 14px;
    background: linear-gradient(90deg, var(--bar-color) 0%, var(--bar-color-light) 100%);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    padding-left: 1rem;
    color: white;
    font-weight: 600;
}

/* Health indicator */
.health-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.2);
    color: white;
    font-weight: 500;
}

.health-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.health-ok {
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
}

.health-error {
    background: #EF4444;
    box-shadow: 0 0 8px #EF4444;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: white;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 1.5rem;
    border: 2px dashed rgba(255, 255, 255, 0.4);
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    color: white;
    font-weight: 500;
}

/* Success/Error messages */
.stSuccess, .stError, .stWarning, .stInfo {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

/* Spinner */
.stSpinner > div {
    border-top-color: white !important;
}
</style>
"""

# ============================================================================
# Helper Functions
# ============================================================================


def check_api_health() -> Dict:
    """Check if the FastAPI server is running and healthy."""
    try:
        response = requests.get(API_HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return {"status": "ok", "data": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def render_health_indicator():
    """Render the API health indicator in the sidebar."""
    health = check_api_health()

    if health["status"] == "ok":
        health_html = """
        <div class="health-indicator">
            <div class="health-dot health-ok"></div>
            <span>API Online</span>
        </div>
        """
        st.sidebar.markdown(health_html, unsafe_allow_html=True)

        data = health["data"]
        st.sidebar.markdown(f"**Device:** {data.get('device', 'unknown')}")
        st.sidebar.markdown(
            f"**Model:** {'Loaded ✓' if data.get('model_loaded') else 'Not loaded ✗'}"
        )
    else:
        health_html = """
        <div class="health-indicator">
            <div class="health-dot health-error"></div>
            <span>API Offline</span>
        </div>
        """
        st.sidebar.markdown(health_html, unsafe_allow_html=True)
        st.sidebar.error(f"Error: {health['message']}")
        st.sidebar.markdown("**Please start the API:**")
        st.sidebar.code("uvicorn src.api:app --reload")


def render_probability_bars(probs: Dict[str, float], top_class: str):
    """Render beautiful animated probability bars."""
    # Sort by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    bars_html = ""
    for class_name, prob in sorted_probs:
        color = CLASS_COLORS.get(class_name, "#6B7280")
        # Lighter version for gradient
        color_light = color + "DD"

        label = CLASS_LABELS.get(class_name, class_name)
        percentage = prob * 100

        # Highlight the top prediction
        highlight = "font-weight: 700; font-size: 1.1rem;" if class_name == top_class else ""

        bars_html += f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">
                <span style="{highlight}">{label} ({class_name})</span>
                <span style="{highlight}">{percentage:.2f}%</span>
            </div>
            <div class="prob-bar" style="--bar-color: {color}; --bar-color-light: {color_light}; width: {percentage}%;">
            </div>
        </div>
        """

    return bars_html


def predict_single_image(image_file) -> Optional[Dict]:
    """Send single image to API for prediction."""
    try:
        files = {"file": (image_file.name, image_file, image_file.type)}
        response = requests.post(API_PREDICT_URL, files=files, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: HTTP {response.status_code}")
            st.error(response.text)
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def predict_batch(
    image_dir: str, report_id: Optional[str], batch_size: int, max_images: Optional[int]
) -> Optional[Dict]:
    """Send batch prediction request to API."""
    try:
        payload = {
            "image_dir": image_dir,
            "batch_size": batch_size,
        }

        if report_id:
            payload["report_id"] = report_id
        if max_images:
            payload["max_images"] = max_images

        response = requests.post(API_BATCH_PREDICT_URL, json=payload, timeout=300)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Batch prediction failed: HTTP {response.status_code}")
            st.error(response.text)
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None


# ============================================================================
# Page: Single Prediction
# ============================================================================


def page_single_prediction():
    """Single image prediction page."""
    st.markdown("<h1>🩹 Single Image Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: white; font-size: 1.1rem;'>Upload a wound image for instant AI classification</p>",
        unsafe_allow_html=True,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, PNG, WEBP",
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<h3>📷 Uploaded Image</h3>", unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("<h3>🔮 Prediction Result</h3>", unsafe_allow_html=True)

            # Predict button
            if st.button("🚀 Classify Image", use_container_width=True):
                with st.spinner("🔄 Running inference..."):
                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Call API
                    result = predict_single_image(uploaded_file)

                    if result:
                        top_class = result["top1_class"]
                        top_prob = result["top1_prob"]
                        all_probs = result["all_probs"]

                        # Main prediction card
                        color = CLASS_COLORS.get(top_class, "#6B7280")
                        label = CLASS_LABELS.get(top_class, top_class)

                        st.markdown(
                            f"""
                            <div class="result-card" style="border-left: 6px solid {color};">
                                <div class="prediction-class" style="color: {color};">
                                    {label}
                                </div>
                                <div class="prediction-prob">
                                    Confidence: {top_prob * 100:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # All probabilities
                        st.markdown("<h3>📊 All Class Probabilities</h3>", unsafe_allow_html=True)
                        prob_bars_html = render_probability_bars(all_probs, top_class)
                        st.markdown(prob_bars_html, unsafe_allow_html=True)

                        # Success message
                        st.success("✅ Prediction completed successfully!")


# ============================================================================
# Page: Batch Prediction
# ============================================================================


def page_batch_prediction():
    """Batch prediction page."""
    st.markdown("<h1>📁 Batch Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: white; font-size: 1.1rem;'>Process multiple images from a directory</p>",
        unsafe_allow_html=True,
    )

    # Input form
    st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        image_dir = st.text_input(
            "Image Directory",
            value="assets/sample_images",
            help="Path to directory containing images (relative or absolute)",
        )

    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=64,
            value=16,
            help="Number of images to process in parallel",
        )

    col3, col4 = st.columns([2, 1])

    with col3:
        report_id = st.text_input(
            "Report ID (optional)",
            value="",
            help="Custom report identifier (leave empty for timestamp)",
        )

    with col4:
        max_images = st.number_input(
            "Max Images (optional)",
            min_value=0,
            value=0,
            help="Limit number of images (0 = no limit)",
        )

    # Run batch prediction
    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        with st.spinner("🔄 Processing images... This may take a while."):
            result = predict_batch(
                image_dir=image_dir,
                report_id=report_id if report_id else None,
                batch_size=batch_size,
                max_images=max_images if max_images > 0 else None,
            )

            if result:
                st.markdown("<h3>📈 Batch Results</h3>", unsafe_allow_html=True)

                # KPI Cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Total Images</div>
                            <div class="kpi-value">{result['total']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-left: 4px solid #10B981;">
                            <div class="kpi-label">Successful</div>
                            <div class="kpi-value" style="color: #10B981;">{result['successful']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="border-left: 4px solid #EF4444;">
                            <div class="kpi-label">Failed</div>
                            <div class="kpi-value" style="color: #EF4444;">{result['failed']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col4:
                    success_rate = (
                        (result["successful"] / result["total"] * 100) if result["total"] > 0 else 0
                    )
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">Success Rate</div>
                            <div class="kpi-value">{success_rate:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Report files
                st.markdown("<h3>📄 Generated Reports</h3>", unsafe_allow_html=True)
                st.markdown(f"**Report Directory:** `{result['report_dir']}`")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**JSONL:** `{Path(result['files']['jsonl']).name}`")

                with col2:
                    st.markdown(f"**CSV:** `{Path(result['files']['csv']).name}`")

                with col3:
                    st.markdown(f"**Per-Image:** `{Path(result['files']['per_image_dir']).name}/`")

                # Failed images
                if result["failed"] > 0:
                    st.markdown("<h3>⚠️ Failed Images</h3>", unsafe_allow_html=True)
                    for failure in result["failed_images"][:10]:  # Show first 10
                        st.error(f"**{failure['image']}**: {failure['error']}")

                    if result["failed"] > 10:
                        st.warning(
                            f"... and {result['failed'] - 10} more failures. Check report for details."
                        )

                # Success message
                st.success("✅ Batch prediction completed!")
                st.balloons()


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main Streamlit application."""
    # Page config
    st.set_page_config(
        page_title="Wound-AI Classification",
        page_icon="🩹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("<h2 style='color: white;'>🩹 Wound-AI</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<p style='color: rgba(255,255,255,0.8);'>Medical Image Classification</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Health indicator
    render_health_indicator()
    st.sidebar.markdown("---")

    # Navigation
    st.sidebar.markdown("<h3 style='color: white;'>Navigation</h3>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "", ["Single Prediction", "Batch Prediction"], label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Info
    st.sidebar.markdown("<h3 style='color: white;'>ℹ️ About</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <p style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
        This application uses a fine-tuned EfficientNet-B0 model to classify wound images
        into 6 categories: Background, D-type, N-type, P-type, S-type, and V-type wounds.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Main content
    if page == "Single Prediction":
        page_single_prediction()
    elif page == "Batch Prediction":
        page_batch_prediction()


if __name__ == "__main__":
    main()
