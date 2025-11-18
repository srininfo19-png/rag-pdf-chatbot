import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_DIR, DATA_DIR, OPENAI_API_KEY


def load_pdfs_from_folder(folder_path: str):
    """Load all PDF files from a folder."""
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
    return docs


def create_or_update_vectorstore():
    """Read PDFs in data/, create or update FAISS vectorstore."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in environment / Streamlit secrets.")

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading PDFs from:", DATA_DIR)
    raw_docs = load_pdfs_from_folder(DATA_DIR)

    if not raw_docs:
        print("No PDFs found in data/ folder.")
        return

    print(f"Loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    docs = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks")

    embeddings = OpenAIEmbeddings()

    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTORSTORE_DIR)
    else:
        # Load existing, merge new docs, and save
        existing_vs = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        new_vs = FAISS.from_documents(docs, embeddings)
        existing_vs.merge_from(new_vs)
        existing_vs.save_local(VECTORSTORE_DIR)

    print("Vectorstore created/updated!")


def get_vectorstore():
    """Load the FAISS vectorstore from disk."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in environment / Streamlit secrets.")

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs
