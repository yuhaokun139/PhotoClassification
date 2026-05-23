import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration
)
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Page config
st.set_page_config(page_title="Grocery Product Classifier", layout="wide")
st.title("🛒 Grocery Product Classifier & Tagging")
st.markdown("Upload a product image...")

# ==========================
# 1. Load grocery classification model (fine-tuned ConvNeXt / ViT / Swin)
@st.cache_resource
def load_grocery_classifier():
    model_id = "facebook/convnext-tiny-224"
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

# 2. Load image captioning model (BLIP) for fine-grained tag generation
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

# 3. Predict product category
def predict_category(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_idx].item()
    label = model.config.id2label[predicted_idx]
    return label, confidence

# 4. Generate detailed description using BLIP
def generate_description(image, blip_processor, blip_model):
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50, num_beams=4)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# ==========================
# Main UI
uploaded_file = st.file_uploader("📁 Choose an image of a grocery product", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Product Image", use_container_width=True)

    # Load models (cached)
    with st.spinner("⏳ Loading..."):
        cls_processor, cls_model = load_grocery_classifier()
        blip_processor, blip_model = load_blip_model()

    # Classify product category
    with st.spinner("🏷️ Recognizing product category..."):
        category, conf = predict_category(image, cls_processor, cls_model)

    # Generate fine-grained description
    with st.spinner("📝 Generating product description..."):
        caption = generate_description(image, blip_processor, blip_model)

    # Show results
    with col2:
        st.success("✅ Analysis complete!")
        
        st.subheader("🏷️ Product Category")
        st.write(f"**{category}**  (confidence: {conf:.2%})")
        
        st.subheader("📖 Product Description")
        st.write(caption)
        
else:
    st.info("👆 Please upload an image of a grocery product to get started.")
