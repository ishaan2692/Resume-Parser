import streamlit as st
import requests
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def generate_text(uploaded_file, job_description):
    try:
        if uploaded_file is not None:
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text:
                st.write("Extracted Text from PDF:")
                st.text_area("Extracted Text", pdf_text, height=300)

        prompt = (
            "Assess candidate fit for the job description. Consider substitutes for skills and experience:\n\n"
            "Skills: Match or equivalent technologies.\n"
            "Experience: Relevance to key responsibilities.\n"
            "Fit: Suitability based on experience and skills.\n\n"
            f"Job Description:\n{job_description}\n\nResume Content:\n{pdf_text}"
        )

        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        with st.spinner('Generating...'):
            response = model.generate_content([prompt], stream=True)
            response.resolve()
            st.markdown(response.text)

    except ValueError:
        error_message = "The provided content might contain hateful or inappropriate elements. Processing failed."
        st.write(f"<div style='padding: 10px; border-radius: 5px;'> {error_message} </div>", unsafe_allow_html=True)

st.set_page_config(page_title="PDF and Job Description Analysis App")

st.sidebar.header("Navigation")
selected_page = st.sidebar.selectbox("Select a page", [
    "Home",
    "Job Description Analysis",
])

if selected_page == "Home":
    st.title("Welcome to Matchify!")
    st.write('"Connecting opportunities with the perfect fit."')
    st.write("Matchify is your go-to tool for seamlessly matching job descriptions with relevant PDF documents. "
             "Simply upload your job descriptions and PDF resumes, and let Matchify do the work. "
             "Our advanced text analysis and similarity matching technology will help you find the best candidates "
             "for your job openings, making the hiring process more efficient and effective.")
    st.write("Created by Ishaaan.")
    st.write("Connect with me:")
    st.markdown("[GitHub](https://github.com/ishaan2692)")
    st.markdown("[LinkedIn](https://in.linkedin.com/in/ishaanbagul)")

elif selected_page == "Job Description Analysis":
    st.header("Job Description Analysis")
    uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])
    job_description = st.text_area("Job Description", "")

    if st.button('Analyze'):
        generate_text(uploaded_file, job_description)
