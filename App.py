import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")  # Load from .env file
genai.configure(api_key=google_api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say "I cannot provide the correct answer as it is not in the context", don't provide a wrong answer.
    Context: \n {context}? \n
    Question: \n {question} \n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Set allow_dangerous_deserialization to True
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Multiple PDF Chatbot")
    st.header("Chat with multiple PDFs using Gemini Pro")

    # Display the sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if pdf_docs and st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

    user_question = st.text_input("Ask a question based on the PDF files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
