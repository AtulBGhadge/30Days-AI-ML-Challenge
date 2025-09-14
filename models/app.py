import streamlit as st
import torch
import os, joblib, pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = "./results"

# --------------------------
# Load Model + Tokenizer
# --------------------------
@st.cache_resource(show_spinner=True)
def load_resources():
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        st.error("❌ Model not found in ./results. Please train and save first.")
        st.stop()

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None

    model.eval()
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_resources()

# --------------------------
# Prediction Function
# --------------------------
@st.cache_data(show_spinner=False)
def predict(texts):
    if isinstance(texts, str):
        texts = [texts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)

    if label_encoder:
        labels = label_encoder.inverse_transform(preds)
        label_names = label_encoder.classes_
    else:
        labels = preds.astype(str)
        label_names = [str(i) for i in range(probs.shape[1])]

    return labels, probs, label_names

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(
    page_title="🌸 Japanese Text Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌸 Japanese Text Classifier")
st.markdown("Classify Japanese sentences into categories such as greetings, food, price, apology, etc.")

tab1, tab2, tab3 = st.tabs(["✍️ Single Sentence", "📑 Multi-line Input", "📂 CSV Upload"])

# --- Single Sentence ---
import re
import plotly.graph_objects as go

# Mapping Japanese categories to English meaning
category_meanings = {
    "greeting": "Greeting",
    "goodbye": "Goodbye",
    "thanks": "Thanks",
    "apology": "Apology",
    "price": "Price inquiry",
    "food": "Food / Drink",
    "location": "Location question"
}

# --------------------------
# Function to split a Japanese line into phrases
# --------------------------
def split_japanese_line(line):
    # Split by common sentence-ending punctuation
    phrases = re.split(r'(。|！|？|\n)', line)
    # Recombine punctuation with text
    phrases = [p1+p2 for p1, p2 in zip(phrases[::2], phrases[1::2])] if len(phrases) > 1 else [line]
    # Remove empty strings and whitespace
    return [p.strip() for p in phrases if p.strip()]

# --- Single Sentence / Multi-phrase Input ---
import re
import plotly.graph_objects as go
import streamlit as st

# --------------------------
# Mapping Japanese categories to English meaning
# --------------------------
category_meanings = {
    "greeting": "Greeting",
    "goodbye": "Goodbye",
    "thanks": "Thanks",
    "apology": "Apology",
    "price": "Price inquiry",
    "food": "Food / Drink",
    "location": "Location question"
}

# --------------------------
# Function to split a Japanese line into phrases
# --------------------------
def split_japanese_line(line):
    # Split by common sentence-ending punctuation
    phrases = re.split(r'(。|！|？|\n)', line)
    # Recombine punctuation with text
    phrases = [p1+p2 for p1, p2 in zip(phrases[::2], phrases[1::2])] if len(phrases) > 1 else [line]
    # Remove empty strings and whitespace
    return [p.strip() for p in phrases if p.strip()]

# --- Single Sentence / Multi-phrase Input ---
with tab1:
    st.subheader("✍️ Enter one or multiple phrases in a single line")
    text_input = st.text_area(
        "Enter Japanese text:",
        "こんにちは これはいくらですか？ 美味しいです ありがとうございます！ ごめんなさい トイレはどこですか？ またね"
    )
    
    if st.button("🔍 Predict", key="single_line_split"):
        if text_input.strip():
            # Split input line into phrases
            phrases = split_japanese_line(text_input.strip())
            labels, probs, label_names = predict(phrases)
            
            st.markdown("### 🔹 Predictions Per Phrase")
            # Display predictions and probability charts for each phrase
            for i, phrase in enumerate(phrases):
                st.markdown(f"**Phrase:** {phrase}  |  **Predicted:** `{labels[i]}`  |  Meaning: *{category_meanings.get(labels[i],'')}*")
                
                fig = go.Figure([go.Bar(
                    x=label_names,
                    y=probs[i],
                    marker_color=probs[i],
                    text=[f"{p:.2f}" for p in probs[i]],
                    textposition="auto"
                )])
                fig.update_layout(
                    title=f"Prediction Confidence for: {phrase}",
                    yaxis=dict(range=[0,1])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # --------------------------
            # Overall meaning of the whole line
            # --------------------------
            overall_meanings = [category_meanings.get(label, "") for label in labels]
            whole_meaning = " | ".join(overall_meanings)
            st.markdown("---")
            st.markdown(f"### 📝 Overall Meaning of the Line / Paragraph:\n{whole_meaning}")
        else:
            st.warning("⚠️ Please enter at least one phrase or sentence.")

# --- Multi-line Input ---
with tab2:
    st.subheader("📑 Enter multiple sentences (one per line)")
    multi_input = st.text_area("Enter Japanese sentences:", "こんにちは\nこれはいくらですか？\n美味しいです")
    if st.button("🚀 Predict All", key="multi"):
        lines = [l.strip() for l in multi_input.split("\n") if l.strip()]
        if lines:
            labels, probs, label_names = predict(lines)
            df = pd.DataFrame({"Text": lines, "Prediction": labels})

            # Show color-coded labels
            for idx, row in df.iterrows():
                st.markdown(f"- **{row['Text']}** → `{row['Prediction']}`")

            st.markdown("### 📊 Confidence Charts")
            for i, line in enumerate(lines):
                fig = px.bar(
                    x=label_names,
                    y=probs[i],
                    labels={"x": "Category", "y": "Confidence"},
                    color=probs[i],
                    color_continuous_scale="Viridis",
                    title=f"Confidence for: {line}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please enter at least one line.")

# --- CSV Upload ---
with tab3:
    st.subheader("📂 Upload CSV (must have a column named `text`)")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("❌ CSV must contain a column named `text`")
        else:
            texts = df["text"].astype(str).tolist()
            labels, probs, _ = predict(texts)
            df["Prediction"] = labels

            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions CSV", csv_bytes, file_name="predictions.csv")

st.markdown("---")
st.markdown("👨‍💻 Built with ❤️ using [Streamlit](https://streamlit.io) & Hugging Face Transformers")
