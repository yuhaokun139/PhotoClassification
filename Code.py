import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification, BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Page config
st.set_page_config(page_title="Short Video Classifier & Tagger", layout="wide")
st.title("🎬 Short Video Classifier & Auto Tagging")
st.markdown("Upload a short video (mp4/avi/mov) – AI will classify the video category and generate a description with tags.")

# ==========================
# 1. Load video classification model (TimeSformer)
def load_video_classifier():
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.eval()
    return processor, model

# 2. Load image captioning model (BLIP)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

# Sample frames uniformly from video
def sample_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# Predict video category using TimeSformer
def predict_video_category(video_frames, processor, model):
    inputs = processor(video_frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_idx].item()
    label = model.config.id2label[predicted_idx]
    return label, confidence

# Generate caption from a single image using BLIP
def generate_description(image, blip_processor, blip_model):
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50, num_beams=4)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Extract simple tags from the generated caption
def extract_tags(caption):
    stopwords = {"a", "an", "the", "of", "to", "and", "in", "is", "are", "was", "were",
                 "this", "that", "these", "those", "for", "with", "on", "at", "by"}
    words = caption.lower().split()
    tags = [w for w in words if w.isalpha() and w not in stopwords and len(w) > 2]
    unique_tags = list(dict.fromkeys(tags))[:8]
    return unique_tags

# ==========================
# Main UI
uploaded_file = st.file_uploader("📁 Choose a short video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(video_path)

    # Load models (cached)
    with st.spinner("⏳ Loading AI models..."):
        vid_processor, vid_model = load_video_classifier()
        blip_processor, blip_model = load_blip_model()

    # Sample frames
    with st.spinner("📸 Analyzing video frames..."):
        frames = sample_frames(video_path, num_frames=8)
        if len(frames) < 8:
            st.error("Not enough frames. Please upload a longer video (at least 1 second).")
            st.stop()

    # Predict category
    with st.spinner("🏷️ Recognizing video category..."):
        category, conf = predict_video_category(frames, vid_processor, vid_model)

    # Generate description (use middle frame)
    mid_frame = frames[len(frames)//2]
    with st.spinner("📝 Generating video description..."):
        caption = generate_description(mid_frame, blip_processor, blip_model)
    tags = extract_tags(caption)

    # Show results
    with col2:
        st.success("✅ Analysis complete!")
        st.subheader("🏷️ Video Category")
        st.write(f"**{category}**  (confidence: {conf:.2%})")
        
        st.subheader("📖 Smart Description")
        st.write(caption)
        
        st.subheader("🔖 Auto Tags")
        st.markdown(", ".join([f"`{tag}`" for tag in tags]))
else:
    st.info("👆 Please upload a short video to get started.")
