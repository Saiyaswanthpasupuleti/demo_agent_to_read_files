from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import os
load_dotenv()

# Google API Key
google_api_key = os.getenv("googleApiKey")

# Document Loader
file_path = "/Users/stackular/Desktop/agent/Python Programming.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()


# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(docs)
print(chunks[0])

# Embeddings - using Google's embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key
)


#  vector db

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # saved to disk automatically
)

print("✅ Vector store created and saved!")

query = "What is a Python Programming?"
results = vector_store.similarity_search(query, k=3)

print("\n🔍 Search Results:")