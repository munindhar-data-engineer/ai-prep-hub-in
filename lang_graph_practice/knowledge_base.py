from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# load the document
loader = PyPDFLoader("resources/vpn_user_guide.pdf")
documents = loader.load()

# split and chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# create embeddings and store in vector store
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("vector_index")
print(f"Knowledge base has been created from {len(documents)} pages, and {len(chunks)} chunks.")