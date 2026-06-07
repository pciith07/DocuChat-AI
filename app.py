import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Config
# -----------------------

load_dotenv()

DB_PATH = "chroma_db"

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------
# Session State
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Models
# -----------------------

embedding_model = MistralAIEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-2506"
)

# -----------------------
# Prompt
# -----------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer cannot be found,
respond:

'I could not find the answer in the document.'
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

# -----------------------
# UI Header
# -----------------------

st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload PDFs, build a knowledge base and chat with your documents."
)

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    build_db = st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    )

    clear_db = st.button(
        "🗑️ Clear Database",
        use_container_width=True
    )

# -----------------------
# Clear Database
# -----------------------

if clear_db:

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    st.session_state.messages = []

    st.success("Database cleared.")

# -----------------------
# Build Vector Database
# -----------------------

if build_db:

    if not uploaded_files:
        st.error("Please upload at least one PDF.")
        st.stop()

    documents = []

    with st.spinner("Processing PDFs..."):

        for pdf in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(pdf.read())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)

            docs = loader.load()

            documents.extend(docs)

            os.remove(temp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(
            documents
        )

        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )

    st.success(
        f"Knowledge Base Created ({len(chunks)} chunks)"
    )

# -----------------------
# Load Vector Store
# -----------------------

vectorstore = None

if os.path.exists(DB_PATH):

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

# -----------------------
# Chat History
# -----------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# Chat Input
# -----------------------

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    if vectorstore is None:

        st.error(
            "Please upload PDFs and build the knowledge base first."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    with st.spinner("Searching Documents..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(
            final_prompt
        )

        answer = response.content

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander(
            "🔍 Retrieved Chunks"
        ):

            for idx, doc in enumerate(
                docs,
                start=1
            ):

                st.markdown(
                    f"### Chunk {idx}"
                )

                st.write(
                    doc.page_content
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )