import streamlit as st
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---- Text Cleaning Function ----
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# ---- Similarity Score Function ----
def calculate_similarity(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0] * 100, 2)

# ---- Streamlit UI ----
st.title("📄 AI Resume & Job Matcher")
st.write("Upload your resume and job description to get a match score!")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
jd_input = st.text_area("Paste Job Description Here", height=200)

if st.button("🔍 Match Resume with JD"):
    if resume_file and jd_input:
        if resume_file.name.endswith(".docx"):
            resume_text = docx2txt.process(resume_file)
        else:
            st.warning("Please upload a DOCX file. PDF not supported in demo.")
            st.stop()

        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(jd_input)

        score = calculate_similarity(cleaned_resume, cleaned_jd)

        st.success(f"✅ Match Score: {score}%")

        if score < 50:
            st.warning("⚠️ Your resume is not closely aligned with the job description. Try adding more relevant keywords.")
        elif score < 75:
            st.info("🔁 Decent match. Consider minor improvements.")
        else:
            st.balloons()
            st.success("🎉 Great match! You're ready to apply.")
    else:
        st.error("Please upload a resume and enter job description.")
