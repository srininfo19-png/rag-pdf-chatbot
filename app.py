import os
import streamlit as st

from config import DATA_DIR, VECTORSTORE_DIR, ADMIN_PASSWORD
from utils.ingest import create_or_update_vectorstore
from utils.rag_chain import get_rag_chain


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None


def load_chain_if_exists():
    """Load RAG chain only if vectorstore folder exists."""
    if os.path.exists(VECTORSTORE_DIR):
        try:
            st.session_state.qa_chain = get_rag_chain()
        except Exception as e:
            st.error(f"Error loading vectorstore: {e}")
    else:
        st.warning("No vectorstore found. Admin must upload PDFs and build the index first.")


def admin_page():
    st.title("🔐 Admin – Manage PDFs & Index")

    password = st.text_input("Enter admin password", type="password")
    if not password:
        st.stop()

    if password != ADMIN_PASSWORD:
        st.error("Incorrect password")
        st.stop()

    st.success("Admin authenticated ✅")

    st.subheader("📂 Upload PDFs (Bulk)")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for pdf in uploaded_files:
            save_path = os.path.join(DATA_DIR, pdf.name)
            with open(save_path, "wb") as f:
                f.write(pdf.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to `{DATA_DIR}`.")

    if st.button("🚀 Build / Update Vector Index"):
        with st.spinner("Building vectorstore from PDFs..."):
            try:
                create_or_update_vectorstore()
                st.success("Vectorstore created/updated successfully!")
            except Exception as e:
                st.error(f"Error while building vectorstore: {e}")

        # Reload chain after update
        load_chain_if_exists()


def user_page():
    st.title("🙋 User – Chat with PDFs")

    if st.session_state.qa_chain is None:
        load_chain_if_exists()

    if st.session_state.qa_chain is None:
        st.warning("No index available. Please ask admin to upload PDFs and build the index.")
        return

    # Show previous messages
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

    st.markdown("---")

    user_query = st.text_input("Ask your question about the PDFs")

    if st.button("Send") and user_query.strip():
        st.session_state.chat_history.append(("user", user_query))

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": user_query})
                answer = result["result"]
                sources = result.get("source_documents", [])
            except Exception as e:
                st.error(f"Error while answering: {e}")
                return

        st.session_state.chat_history.append(("bot", answer))
        st.markdown(f"**🤖 Bot:** {answer}")

        with st.expander("🔍 Show Sources"):
            if not sources:
                st.write("No sources found.")
            else:
                for idx, doc in enumerate(sources):
                    st.markdown(f"**Source {idx+1}:** `{doc.metadata.get('source', 'unknown')}`")
                    st.write(doc.page_content[:500] + "...")


def main():
    # Make sure folders exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    init_session_state()

    st.sidebar.title("RAG PDF Chatbot")
    role = st.sidebar.radio("Choose Mode", ["User", "Admin"])

    if role == "Admin":
        admin_page()
    else:
        user_page()


if __name__ == "__main__":
    main()
