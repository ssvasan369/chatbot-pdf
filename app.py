import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Custom styles for improved UI/UX
def apply_custom_styles():
    st.markdown(
        """
        <style>
        .main-header {
            font-family: 'Arial Black', sans-serif;
            font-size: 2.5rem;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-family: 'Roboto', sans-serif;
            font-size: 1.5rem;
            color: #888;
            margin-bottom: 10px;
        }
        .file-uploader {
            border: 2px dashed #4CAF50;
            padding: 10px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Processing PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Splitting text into small chunks to create embeddings
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Using Google's embedding004 model to create embeddings and FAISS to store the embeddings
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Handling user questions
def handle_userinput(question):
    response = st.session_state.conversation.invoke({"question": question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)  # Return only the answer from the response

# Storing conversations as chain of outputs
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")

    apply_custom_styles()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main header
    st.markdown("<h1 class='main-header'>Chat with Multiple PDFs :books:</h1>", unsafe_allow_html=True)

    # User input for question
    st.markdown("<h2 class='sub-header'>Ask a question about your documents:</h2>", unsafe_allow_html=True)
    user_question = st.text_input("Type your question here...", placeholder="e.g., What is the summary of document X?")

    if user_question:
        handle_userinput(user_question)

    # Sidebar for uploading files
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Upload Your Documents</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Drag and drop your PDFs here:",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files to process.",
        )

        if st.button("Process"):
            with st.spinner("Processing your documents. Please wait..."):
                raw_text = get_pdf_text(pdf_docs)

                # Convert to chunks
                text_chunks = get_text_chunks(raw_text)
                st.write("Document has been split into smaller chunks for processing.")

                # Create embeddings
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
