import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Supported language mapping (you can add more as needed)
language_codes = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh",
    "Arabic": "ar",
    "Russian": "ru",
    "Japanese": "ja",
    "Bengali": "bn",
    "Tamil": "ta",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Danish": "da",
    "Dutch": "nl",
    "Finnish": "fi",
    "Gujarati": "gu",
    "Italian": "it",
    "Punjabi": "pa",
    "Marathi": "mr"
}

@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def translate_m2m100(text, source_lang_code, target_lang_code):
    tokenizer.src_lang = source_lang_code
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang_code))
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Streamlit UI
st.title("🌍 AI-Powered Multi-Language Translator")
st.write("Using Facebook's M2M100 Multilingual Model")

text = st.text_area("Enter the text to translate:")
source_lang = st.selectbox("Translate from:", list(language_codes.keys()))
target_lang = st.selectbox("Translate to:", list(language_codes.keys()))

if st.button("Translate"):
    if source_lang == target_lang:
        st.warning("Please select different source and target languages.")
    elif not text.strip():
        st.warning("Please enter some text to translate.")
    else:
        src_code = language_codes[source_lang]
        tgt_code = language_codes[target_lang]
        translated_text = translate_m2m100(text, src_code, tgt_code)
        st.success(f"**Translated Text:** {translated_text}")
