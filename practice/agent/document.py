from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb
from uuid import uuid4
load_dotenv()

################################################################################## 
# Document Loader
file_path="/Users/stackular/Desktop/practice/projectfiles/Python Programming.pdf"
loader=PyPDFLoader(file_path)
document=loader.load()
################################################################################## 

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
texts = text_splitter.split_documents(document)
texts_content = [t.page_content for t in texts]

################################################################################## 
# Embeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vectors = embeddings.embed_documents(texts_content)

################################################################################## 

client =chromadb.Client()

colection= client.create_collection('my_agent_collection')

colection.add(
    ids=[str(uuid4()) for _ in texts],
    documents=texts_content,
    embeddings=vectors
)

results= colection.query(
    query_texts=["What is the main topic of the document?"],
    n_results=3,
 )

print(results)