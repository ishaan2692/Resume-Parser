import streamlit as st
import os
import pdfplumber  # New library for PDF text extraction
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Function to extract text from PDFs in a folder using pdfplumber
def extract_text_from_pdfs(folder_path):
    pdf_texts = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        pdf_texts.append((pdf_file, text))
    return pdf_texts

# Function to calculate the similarity between the job description and the PDF content
def find_best_match(job_description, pdf_texts):
    # Use TF-IDF to vectorize the texts
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Prepare the documents (job description + PDFs)
    documents = [job_description] + [text for filename, text in pdf_texts]
    
    # Vectorize the documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity between the job description (first item) and all PDFs
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get the most similar PDF
    best_match_idx = np.argmax(cosine_similarities)
    best_match_filename, best_match_text = pdf_texts[best_match_idx]
    
    return best_match_filename, cosine_similarities[best_match_idx]

# Streamlit interface
st.set_page_config(page_title="PDF Job Description Matcher")

st.sidebar.header("Navigation")
selected_page = st.sidebar.selectbox("Select a page", ["Home", "Job Description Matcher"])

if selected_page == "Home":
    st.title("Welcome to the Job Description Matcher!")
    st.write("This app allows you to find the best matching job descriptions from a set of PDFs.")

elif selected_page == "Job Description Matcher":
    st.header("Job Description Matcher")
    
    # Allow user to upload a folder of PDFs
    uploaded_folder = st.text_input("Enter the folder path containing PDFs:")
    job_description = st.text_area("Enter the job description:")
    
    if st.button("Match Job Description"):
        if uploaded_folder and job_description:
            try:
                # Extract text from PDFs in the folder
                pdf_texts = extract_text_from_pdfs(uploaded_folder)
                
                # Find the best match
                best_match_filename, similarity_score = find_best_match(job_description, pdf_texts)
                
                # Show results
                st.write(f"Best matching PDF: {best_match_filename}")
                st.write(f"Similarity Score: {similarity_score * 100:.2f}%")
                st.write(f"**Excerpt from the matched document:**")
                st.write(pdf_texts[[filename for filename, _ in pdf_texts].index(best_match_filename)][1][:500])  # Show first 500 chars of matched text
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please provide both the folder path and a job description.")
