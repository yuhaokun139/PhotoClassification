import streamlit as st
from transformers import pipeline

# ============================================================
# Configuration
# ============================================================
# Replace with your fine-tuned model ID on Hugging Face Hub
CLASSIFICATION_MODEL_ID = "yuhaokun/distilbert-ag-news"

# NER model for company name extraction (specialized for company names)
NER_MODEL_ID = "nbroad/deberta-v3-base-company-names"

# ============================================================
# Cache model loaders
# ============================================================
@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model=CLASSIFICATION_MODEL_ID,
        device=-1,          # CPU (Streamlit Cloud has no GPU)
        truncation=True,
        max_length=512
    )

@st.cache_resource
def load_ner():
    return pipeline(
        "token-classification",
        model=NER_MODEL_ID,
        aggregation_strategy="simple",
        device=-1
    )

# ============================================================
# Helper functions
# ============================================================
def map_label(label):
    """
    Map the model's output label to AG News category names.
    Supports both 'LABEL_0' format and direct category names.
    """
    label_mapping = {
        "LABEL_0": "World",
        "LABEL_1": "Sports",
        "LABEL_2": "Business",
        "LABEL_3": "Sci/Tech"
    }
    if label in label_mapping:
        return label_mapping[label]
    # Direct output like 'World' (some fine-tuned models)
    if label in ["World", "Sports", "Business", "Sci/Tech"]:
        return label
    return label  # fallback

def extract_company_names(ner_output):
    """
    Extract company names from NER pipeline output.
    Handles different field names ('entity_group', 'entity', 'type')
    and entity types ('COMPANY', 'ORG', 'B-COMPANY', 'I-COMPANY', etc.)
    """
    companies = []
    for entity in ner_output:
        # Get entity type from various possible keys
        ent_type = entity.get('entity_group') or entity.get('entity') or entity.get('type') or ''
        ent_type = ent_type.upper()
        # Accept common company/organization labels
        if ent_type in ('COMPANY', 'ORG', 'B-COMPANY', 'I-COMPANY', 'B-ORG', 'I-ORG'):
            word = entity.get('word', '')
            # Some models add leading '▁' or '##' - clean them
            if word.startswith('▁'):
                word = word[1:]
            if word.startswith('##'):
                word = word[2:]
            if word:
                companies.append(word)
    # Remove duplicates while preserving order
    seen = set()
    unique_companies = []
    for c in companies:
        if c not in seen:
            seen.add(c)
            unique_companies.append(c)
    return unique_companies

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="News Classifier & Company Extractor", layout="centered")
st.title("📰 News Classifier & Company Extractor")
st.markdown("Enter a news article to see its category and mentioned companies.")

user_input = st.text_area("News article text:", height=200)

if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            classifier = load_classifier()
            ner_pipeline = load_ner()

            # ---- Classification ----
            cls_result = classifier(user_input, truncation=True, max_length=512)[0]
            raw_label = cls_result['label']
            confidence = cls_result['score']
            category = map_label(raw_label)

            # ---- NER (Company extraction) ----
            ner_output = ner_pipeline(user_input)
            companies = extract_company_names(ner_output)

            # ---- Display Results ----
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
                    st.info("No company names detected. The NER model may not have recognized any company entity in this text.")
