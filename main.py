import streamlit as st
import pymupdf
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Configuration and Setup
st.set_page_config(page_title="Finance Analyzer", page_icon="📊", layout="wide")

# Load environment variables from the .env file
load_dotenv()



def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF
    """
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text):
    """
    Split text into manageable chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    """
    Create vector store for semantic search
    """
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def retrieve_relevant_chunks(vector_store, query, k=5):
    """
    Retrieve most relevant chunks for a query
    """
    return vector_store.similarity_search(query, k=k)

def generate_response_with_groq(query, context):
    """
    Generate response using Groq Llama 3.8 LPU
    """
    chat = ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Combine context and query
    context_str = " ".join([chunk.page_content for chunk in context])
    prompt = f"""
    Context: {context_str}
    
    Question: {query}
    
    Please provide a precise, informative answer based on the given context. 
    If the answer is not in the context, state that you cannot find the information.
    """
    
    response = chat.invoke(prompt)
    return response.content

def main():
    st.title("Finance Analyzer")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Fianace Statement/Budget PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        # Add some styling
        st.markdown("""
        <style>
        .reportview-container {
            background: linear-gradient(to right, #f0f2f6, #e6e9ef);
        }
        .sidebar .sidebar-content {
            background: rgba(255,255,255,0.7);
        }
        </style>
        """, unsafe_allow_html=True)

    # Main content area
    if uploaded_file is not None:
        # Save uploaded file
        with open("budget.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract and process text
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf("budget.pdf")
            chunks = chunk_text(text)
            vector_store = create_vector_store(chunks)
        
        # Query interface
        st.header("Ask Questions about the Budget")
        query = st.text_input("Enter your question:", 
                               placeholder="What are the key economic initiatives?")
        
        if query:
            with st.spinner("Generating response..."):
                # Retrieve relevant context
                context = retrieve_relevant_chunks(vector_store, query)
                
                # Generate response
                response = generate_response_with_groq(query, context)
                
                # Display response
                st.markdown("### 📝 Analysis")
                st.write(response)
                
                # Show retrieved context chunks
                with st.expander("Retrieved Context"):
                    for chunk in context:
                        st.write(chunk.page_content)

    else:
        st.info("Please upload a PDF of the Indian Budget")

# Run the app
if __name__ == "__main__":
    main()