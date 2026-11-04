import streamlit as st
from transformers import pipeline

st.title("BERT vs GPT-2")

@st.cache_resource
def load_bert():
    return pipeline("text-classification",
                    model="bhadresh-savani/bert-base-uncased-emotion")

@st.cache_resource
def load_gpt():
    return pipeline("text-generation", model="gpt2")

tab1, tab2 = st.tabs(["BERT (Encoder)", "GPT-2 (Decoder)"])

with tab1:
    st.subheader("Part A: Emotion classification")
    text = st.text_input("Enter a sentence:", "I am so happy today!")
    if st.button("Classify"):
        out = load_bert()(text)[0]
        st.write("Emotion:", out["label"])
        st.write("Confidence:", round(out["score"], 3))

with tab2:
    st.subheader("Part B: Text generation")
    start = st.text_input("Start of a sentence:", "Artificial intelligence will")
    if st.button("Generate"):
        out = load_gpt()(start, max_new_tokens=30, do_sample=True)
        st.write(out[0]["generated_text"])
