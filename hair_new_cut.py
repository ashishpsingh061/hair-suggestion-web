import streamlit as st
import base64
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

# Set up the page configuration
st.set_page_config(page_title="AI Hairstyle Suggestion", layout="wide")

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Replace with the path to your logo image
logo_path = "C:/Users/DELL/Pictures/Saved Pictures/download (1).png"
logo_base64 = get_base64_image(logo_path)

header_html = f'''
<div style="display: flex; align-items: center; justify-content: center; background-color: #005580; padding: 10px; border-radius: 5px;">
    <h1 style="color: white; margin: 0;">AI Hairstyle Suggestion</h1>
    <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 50px; margin-left: 10px;">
</div>
'''
# Display the header in Streamlit
st.markdown(header_html, unsafe_allow_html=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Predefined hairstyle suggestions
HAIRSTYLE_SUGGESTIONS = {
    "oval": ["Long Layers", "Wavy Bob", "Side-Swept Bangs", "Classic Pixie Cut"],
    "round": ["Layered Bob", "Choppy Pixie Cut", "Side-Parted Long Waves", "Face-Framing Layers"],
    "square": ["Soft Curls", "Curtain Bangs", "Feathered Layers", "Shoulder-Length Waves"]
}

def detect_face(image):
    """Detects face and returns bounding box."""
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            img_h, img_w, _ = image_rgb.shape
            x, y, w, h = (
                int(bboxC.xmin * img_w),
                int(bboxC.ymin * img_h),
                int(bboxC.width * img_w),
                int(bboxC.height * img_h),
            )
            return (x, y, w, h)
    return None

def classify_face_shape(image):
    """ Simple heuristic-based face shape classification. """
    width, height = image.size
    aspect_ratio = width / height

    print('aspect_ratio',aspect_ratio)
    if aspect_ratio > 0.9 and aspect_ratio < 1.1:
        return "round"
    elif aspect_ratio > 1.2:
        return "square"
    else:
        return "oval"

def image_to_bytes(image):
    """Converts PIL image to bytes."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def main():
    st.markdown('<h1 class="title">Welcome to the AI Hairstyle Suggestion App</h1>', unsafe_allow_html=True)

    # File Uploader
    with st.container():
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your face image and get text-based hairstyle recommendations!", type=["jpg", "png", "jpeg"], key="file_uploader")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        with st.container():
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        face_coords = detect_face(image)
        print('face_coords',face_coords)

        if face_coords:
            st.success("✅ Face Detected!")

            face_shape = classify_face_shape(image)
            st.info(f"Detected Face Shape: **{face_shape.capitalize()}**")

            # Fetch hairstyle suggestions
            hairstyle_suggestions = HAIRSTYLE_SUGGESTIONS.get(face_shape, [])

            if hairstyle_suggestions:
                with st.container():
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    st.subheader("💇 Recommended Hairstyles:")
                    for style in hairstyle_suggestions:
                        st.markdown(f"- {style}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Convert image to bytes for comparison
                image_bytes = image_to_bytes(image)

                # Check for duplicates before appending
                if not any(np.array_equal(image_bytes, image_to_bytes(entry['image'])) for entry in st.session_state.history):
                    # Save the current image and suggestions to history
                    st.session_state.history.append({
                        'image': image,
                        'face_shape': face_shape,
                        'suggestions': hairstyle_suggestions
                    })

            else:
                st.error("❌ No hairstyle suggestions available.")
        else:
            st.warning("⚠️ No face detected. Please upload a clearer image.")

    # Button to display history
    if st.button("Show History"):
        if st.session_state.history:
            st.markdown("## History")
            cols = st.columns(4)
            for idx, entry in enumerate(st.session_state.history):
                with cols[idx % 4]:
                    st.image(entry['image'], caption=f"Face Shape: {entry['face_shape'].capitalize()}", use_container_width=True)
                    with st.expander("View Suggestions"):
                        for suggestion in entry['suggestions']:
                            st.markdown(f"- {suggestion}")
        else:
            st.info("No history available.")

    # Footer
    st.markdown('<div class="footer">© 2025 AI Hairstyle Suggestion. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()