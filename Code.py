import streamlit as st
import tempfile
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Page config
st.set_page_config(page_title="Image Classifier & Object Tagger", layout="wide")
st.title("🖼️ Image Classifier & Object Tagger")
st.markdown("Upload an image – AI will classify the main subject and generate object labels with descriptions.")

# ==========================
# 1. Load image classification model (ConvNeXt / ViT)
@st.cache_resource
def load_image_classifier():
    processor = AutoImageProcessor.from_pretrained("facebook/convnext-large-224")
    model = AutoModelForImageClassification.from_pretrained("facebook/convnext-large-224")
    model.eval()
    return processor, model

# 2. Load image captioning model (BLIP)
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

# 3. Predict image category (object classification)
def predict_image_category(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_idx].item()
    label = model.config.id2label[predicted_idx]
    return label, confidence

# 4. Generate detailed object description using BLIP
def generate_description(image, blip_processor, blip_model):
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50, num_beams=4)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# 5. Extract simple tags from the generated caption
def extract_tags(caption):
    stopwords = {"a", "an", "the", "of", "to", "and", "in", "is", "are", "was", "were",
                 "this", "that", "these", "those", "for", "with", "on", "at", "by", "behind", "front"}
    words = caption.lower().split()
    tags = [w for w in words if w.isalpha() and w not in stopwords and len(w) > 2]
    unique_tags = list(dict.fromkeys(tags))[:8]
    return unique_tags

# ==========================
# Main UI
uploaded_file = st.file_uploader("📁 Choose an image file", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load models (cached)
    with st.spinner("⏳ Loading..."):
        cls_processor, cls_model = load_image_classifier()
        blip_processor, blip_model = load_blip_model()

    # Classify main object
    with st.spinner("🏷️ Recognizing..."):
        category, conf = predict_image_category(image, cls_processor, cls_model)

    # Generate description and tags
    with st.spinner("📝 Generating object description..."):
        caption = generate_description(image, blip_processor, blip_model)
    tags = extract_tags(caption)

    # Show results
    with col2:
        st.success("✅ Analysis complete!")
        st.subheader("🏷️ Main Object Category")
        st.write(f"**{category}**  (confidence: {conf:.2%})")
        
        st.subheader("📖 Object Description")
        st.write(caption)
        
        st.subheader("🔖 Extracted Tags")
        st.markdown(", ".join([f"`{tag}`" for tag in tags]))
else:
    st.info("👆 Please upload an image to get started.")
