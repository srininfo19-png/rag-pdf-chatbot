import os

# Folders to store data and vector index
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

# API key and admin password will come from environment / Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
