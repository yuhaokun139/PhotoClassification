import streamlit as st
from transformers import pipeline

# ============================================================
# Cache models to avoid reloading on each interaction
# ============================================================
@st.cache_resource
def load_classifier():
    model_id = "yuhaokun/distilbert-ag-news"
    return pipeline("text-classification", model=model_id, device=-1)  # device=-1 uses CPU

@st.cache_resource
def load_ner():
    model_id = "nbroad/deberta-v3-base-company-names"
    return pipeline("token-classification", model=model_id, device=-1, aggregation_strategy="simple")

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="News Classifier & Company Extractor", layout="centered")
st.title("📰 News Classifier & Company Extractor")
st.markdown("Enter an English news article to see its category and mentioned companies.")

# Input text area
user_input = st.text_area("News article text:", height=200)

# Button to trigger analysis
if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Load models
            classifier = load_classifier()
            ner = load_ner()
            
            # Run classification
            cls_result = classifier(user_input, truncation=True, max_length=512)[0]
            label = cls_result['label']
            confidence = cls_result['score']
            
            # Map label to category name (adjust mapping if your fine-tuned model uses different output)
            # AG News standard mapping: 0=World,1=Sports,2=Business,3=Sci/Tech
            # The pipeline may return 'LABEL_0', 'LABEL_1', etc.
            label_map = {
                "LABEL_0": "World",
                "LABEL_1": "Sports",
                "LABEL_2": "Business",
                "LABEL_3": "Sci/Tech"
            }
            category = label_map.get(label, label)
            
            # Run NER for company names
            entities = ner(user_input)
            companies = [ent['word'] for ent in entities if ent.get('entity_group') == 'COMPANY']
            # Remove duplicates while preserving order
            companies = list(dict.fromkeys(companies))
            
            # Display results
            st.subheader("📊 Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("News Category", category)
                st.caption(f"Confidence: {confidence:.2f}")
            with col2:
                if companies:
                    st.write("**Companies found:**")
                    for c in companies:
                        st.write(f"- {c}")
                else:
                    st.info("No company name detected.")
