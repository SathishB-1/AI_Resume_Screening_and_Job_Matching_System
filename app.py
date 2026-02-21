import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Screening & Job Matching System")
st.markdown("Match your resume with real job listings using NLP")

# ---------------------------
# Load Model & Data (Cache for speed)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_pickle("job_data (1).pkl")
    embeddings = np.load("job_embeddings (1).npy")
    return df, embeddings

model = load_model()
df, job_embeddings = load_data()

# ---------------------------
# Resume Input Section
# ---------------------------
st.sidebar.header("📄 Upload Resume")

resume_option = st.sidebar.radio(
    "Choose Resume Input Method:",
    ["Paste Resume Text", "Upload PDF"]
)

resume_text = ""

if resume_option == "Paste Resume Text":
    resume_text = st.text_area("Paste your resume here:", height=300)

else:
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

# ---------------------------
# Process Resume
# ---------------------------
if st.button("🔍 Match Jobs"):

    if resume_text.strip() == "":
        st.warning("Please provide resume text.")
    else:

        with st.spinner("Analyzing Resume..."):

            # Encode Resume
            resume_embedding = model.encode([resume_text])

            # Compute Similarity
            similarities = cosine_similarity(
                resume_embedding,
                job_embeddings
            )[0] * 100

            df['match_score'] = similarities

            top_jobs = df.sort_values(
                by='match_score',
                ascending=False
            ).head(5)

        st.success("Top Matching Jobs Found!")

        # ---------------------------
        # Display Results
        # ---------------------------
        for i, row in top_jobs.iterrows():

            st.markdown("---")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(row['job_title'])
                st.write(f"🏢 {row['organization']}")
                st.write(f"📍 {row['location']}")

            with col2:
                st.metric(
                    label="Match Score",
                    value=f"{row['match_score']:.2f}%"
                )

            st.progress(int(row['match_score']))

        st.markdown("---")
        st.info("Higher score means stronger relevance to your resume.")
