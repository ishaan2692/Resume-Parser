import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import pdfplumber  # Library for PDF text extraction

load_dotenv()

# Configure your Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = "https://your-gemini-api-endpoint"  # Replace with your actual API endpoint

# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(uploaded_pdf):
    with pdfplumber.open(uploaded_pdf) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to calculate similarity using Gemini API
def find_best_match(job_description, pdf_texts):
    best_match_filename = None
    best_similarity_score = 0.0

    for filename, text in pdf_texts:
        # Create the payload for the Gemini API
        payload = {
            "job_description": job_description,
            "pdf_content": text
        }

        # Send request to Gemini API
        response = requests.post(GEMINI_API_URL, headers={"Authorization": f"Bearer {GEMINI_API_KEY}"}, json=payload)

        if response.status_code == 200:
            data = response.json()
            similarity_score = data['similarity_score']  # Adjust based on your API response structure

            # Update best match if the current one is better
            if similarity_score > best_similarity_score:
                best_similarity_score = similarity_score
                best_match_filename = filename
        else:
            st.error("Error in API response: " + response.text)

    return best_match_filename, best_similarity_score

# Streamlit interface
st.set_page_config(page_title="Matchify - PDF Job Description Matcher")

st.sidebar.header("Navigation")
selected_page = st.sidebar.selectbox("Select a page", ["Home", "Job Description Matcher"])

if selected_page == "Home":
    st.title("Welcome to Matchify!")
    st.write('"Connecting opportunities with the perfect fit."')
    st.write("Matchify is your go-to tool for seamlessly matching job descriptions with relevant PDF documents. "
             "Simply upload your job descriptions and PDF resumes, and let Matchify do the work. "
             "Our advanced text analysis and similarity matching technology will help you find the best candidates "
             "for your job openings, making the hiring process more efficient and effective.")
    
    # Add links to GitHub and LinkedIn
    st.write("Connect with me:")
    st.markdown("[GitHub](https://github.com/ishaan2692)")
    st.markdown("[LinkedIn](https://in.linkedin.com/in/ishaanbagul)")

elif selected_page == "Job Description Matcher":
    st.header("Job Description Matcher")
    
    # Allow user to upload multiple PDF files
    uploaded_pdfs = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    job_description = st.text_area("Enter the job description:")
    
    if st.button("Match Job Description"):
        if uploaded_pdfs and job_description:
            try:
                # Extract text from the uploaded PDFs
                pdf_texts = []
                for uploaded_pdf in uploaded_pdfs:
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    pdf_texts.append((uploaded_pdf.name, pdf_text))  # Store file name with text
                
                # Find the best match using the Gemini API
                best_match_filename, similarity_score = find_best_match(job_description, pdf_texts)
                
                # Show results
                if best_match_filename:
                    st.write(f"Best matching PDF: {best_match_filename}")
                    st.write(f"Similarity Score: {similarity_score * 100:.2f}%")
                    matched_text = next(text for filename, text in pdf_texts if filename == best_match_filename)
                    st.write(f"**Excerpt from the matched document:**")
                    st.write(matched_text[:500])  # Show first 500 chars of matched text
                else:
                    st.write("No suitable match found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please provide both the PDFs and a job description.")
