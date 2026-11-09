import streamlit as st
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = "" # paste generated_key from platform.openai.com

st.header("PDF Reader Chatbot")

with st.sidebar:
    st.title("Documents")
    file = st.file_uploader("Upload a file and start asking Questions", type="pdf")

# Extract content from the file and process
if file is not None:
# extract text from the file
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text+=page.extract_text() + "\n"
    st.write(text)

    # Split the content to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", ". ", "\n", " ", ""],
        chunk_size = 800,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        openai_api_key = OPENAI_API_KEY
    )
    # store embeddings in vector
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get use question
    user_question = st.text_input("Type your question...")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # generate answer
    # question -> embeddings -> similarity search -> results to LLM -> response (CHAIN)
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":4}
    )

    # define the llm and provide prompts

    llm = ChatOpenAI(
        model = 'gpt-4o-mini',
        temperature = 0.3,
        max_tokens = 1000,
        openai_api_key = OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a assistant who can read the content from PDF and provide answers to the questions\n"
         "1. Provide well-explained answers using the PDF provided.\n"
         "2. include relevant details and responses\n"
         "3. Only use the content from provided documents\n"
         "4. Summarize log information\n"
         "5. if the information not available in the document, answer the same politely.\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)