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

        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        with st.spinner('Generating...'):
            response = model.generate_content([job_description, pdf_text], stream=True)
            response.resolve()
            st.markdown(response.text)

    except ValueError:
        error_message = "The provided content might contain hateful or inappropriate elements. Processing failed."
        st.write(f"<div style='padding: 10px; border-radius: 5px;'> {error_message} </div>", unsafe_allow_html=True)

def chatbot(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except ValueError as e:
        return f"I apologize, I'm currently encountering some issues and cannot process your request. " \
               f"The error message is: {str(e)}."
    except Exception as e:
        return f"An unexpected error occurred. Please try again later. (Error: {str(e)})"

st.set_page_config(page_title="PDF and Job Description Analysis App")

st.sidebar.header("Navigation")
selected_page = st.sidebar.selectbox("Select a page", [
    "Home",
    "Job Description Analysis",
    "Chatbot",
    "About",
])

if selected_page == "Home":
    st.title("Welcome to our PDF and Job Description Analysis App!")

    html_temp = f"""
    <div style="text-align: center">
      <img src="https://cdn3.emoji.gg/emojis/9228-kiwicatrun.gif" width="200" />
    </div>
    """
    st.write(html_temp, unsafe_allow_html=True)

    st.write("This app offers functionalities to help you with:")
    st.write("- Analyzing job descriptions from PDF files")
    st.write("- Engaging in conversations with a chatbot")
    st.write("- Exploring research papers on relevant topics") 

elif selected_page == "Job Description Analysis":
    st.header("Job Description Analysis")
    uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])
    job_description = st.text_area("Job Description", "Enter the job description text here...")

    if st.button('Analyze'):
        generate_text(uploaded_file, job_description)

elif selected_page == "Chatbot":
    st.header("Chatbot")
    user_input = st.text_input("Enter your message:")

    if user_input:
        bot_response = chatbot(user_input)
        st.write("Chatbot:", bot_response)

elif selected_page == "About":
    st.title("About")
    st.write("""
    This application is designed to assist users in analyzing job descriptions provided in PDF format, 
    as well as to provide a chatbot for conversational interactions. 
    We aim to leverage AI to enhance user experience and provide insightful feedback.
    """)
