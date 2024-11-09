import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Retrieve the Google API key securely from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Step 1: Extract and decrypt PDF text, track page numbers
def get_pdf_text(pdf_docs):
    pages_text = []
    for pdf_data in pdf_docs:
        pdf_file = io.BytesIO(pdf_data.read())  # Convert bytes to file-like object
        pdf_reader = PdfReader(pdf_file)
        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt("")  # Try to decrypt if encrypted
            except Exception as e:
                st.error(f"Failed to decrypt PDF: {e}")
                return []
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append({"text": page_text, "page_num": page_num})
    return pages_text

# Step 2: Split text into chunks and keep track of page numbers
def get_text_chunks(pages_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_with_pages = []
    for page in pages_text:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks_with_pages.append({"text": chunk, "page_num": page["page_num"]})
    return chunks_with_pages

# Step 3: Create or load a vector store with error handling
def load_or_create_vector_store(text_chunks):
    texts = [chunk["text"] for chunk in text_chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    try:
        vector_store = FAISS.from_texts(texts, embedding=embeddings)
    except Exception as e:  # General exception handling for vector store creation
        st.error(f"Error during embedding or creating vector store: {str(e)}")
        return None
    return vector_store

# Step 4: Create the conversational chain
def get_conversational_chain():
    prompt_template = """
    Provide a detailed answer to the question based on the context provided. Elaborate as much as possible and include any relevant details.

    Context:\n{context}\n
    Question:\n{question}\n
    Detailed Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Step 5: User input and generate detailed response with page numbers
def user_input(user_question, chain, vector_store, text_chunks):
    docs = vector_store.similarity_search(user_question)  # Retrieve relevant chunks
    matching_chunks = []
    
    # Identify the page number for each chunk found
    for doc in docs:
        chunk_index = [i for i, chunk in enumerate(text_chunks) if chunk["text"] == doc.page_content]
        if chunk_index:
            matching_chunks.append({
                "text": doc.page_content,
                "page_num": text_chunks[chunk_index[0]]["page_num"]
            })

    # Use the conversational chain to generate a detailed answer based on context
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Build response with page numbers
    detailed_response = response['output_text'] + "\n\n"
    for chunk in matching_chunks:
        detailed_response += f"(Found on page {chunk['page_num']})\n\n{chunk['text']}\n\n"

    return detailed_response

# Main function for loading PDF, processing it, and answering questions
def main():
    # Display the title of the app
    st.title("User Manual ChatBot")

    # Upload PDF file using Streamlit's file uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF text
        pdf_docs = [uploaded_file]  # The uploaded file is in bytes format
        pages_text = get_pdf_text(pdf_docs)
        
        if not pages_text:
            st.error("No text extracted from the PDF. Please check the content.")
            return

        # Split text into chunks with page numbers
        text_chunks = get_text_chunks(pages_text)

        # Load or create vector store
        vector_store = load_or_create_vector_store(text_chunks)
        if vector_store is None:
            return

        # Initialize conversational chain
        chain = get_conversational_chain()

        # Input loop for user questions
        st.write("You can now start asking questions about the PDF content:")
        user_question = st.text_input("Enter your question:")
        
        if user_question:
            # Generate and display a detailed response with page numbers
            answer = user_input(user_question, chain, vector_store, text_chunks)
            st.text_area("Detailed Answer", value=answer, height=300)

# Run the main function
if __name__ == "__main__":
    main()
