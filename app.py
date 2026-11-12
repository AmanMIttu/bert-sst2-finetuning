"""Optional Streamlit interface for local BERT SST-2 inference."""
import streamlit as st

from src.predict import DEFAULT_MODEL_DIR, load_model, predict_sentiment

st.set_page_config(page_title="BERT Sentiment Analysis", page_icon="🎬", layout="centered")
st.title("BERT Movie Review Sentiment")
st.caption("A BERT-base-uncased model fine-tuned on the GLUE SST-2 movie-review dataset.")
st.write("Enter an English movie review to estimate its sentiment.")

with st.expander("Try a sample review"):
    st.markdown("**Positive:** A moving story with warm performances and a wonderful ending.")
    st.markdown("**Negative:** A predictable plot, clumsy dialogue, and a painfully slow pace.")
    st.markdown("**Mixed:** The acting is excellent, but the story never quite comes together.")

@st.cache_resource(show_spinner="Loading the fine-tuned BERT model…")
def get_model():
    return load_model(DEFAULT_MODEL_DIR)

text = st.text_area(
    "Movie review",
    placeholder="The performances were excellent and the story was moving.",
    height=140,
)
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter a movie review before analyzing sentiment.")
    else:
        try:
            model, tokenizer = get_model()
            result = predict_sentiment(text, model=model, tokenizer=tokenizer)[0]
            st.subheader(f"{result['label']} sentiment")
            st.caption("Class probabilities are model estimates, not guarantees or calibrated confidence intervals.")
            negative, positive = st.columns(2)
            negative.metric("Negative", f"{result['probabilities']['Negative']:.1%}")
            positive.metric("Positive", f"{result['probabilities']['Positive']:.1%}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

st.divider()
st.caption("This model was trained on short English movie reviews. It may be less reliable for sarcasm, other topics, or other languages.")
