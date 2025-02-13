import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

# For local testing
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


# Creating the embeddings
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')


# Creating the LLM
llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
openai_llm = ChatOpenAI(model='gpt-4o')


# Streamlit UI Customization (Hogwarts theme)
st.set_page_config(page_title="IntelliSQR Internship Assignment", page_icon="🧠")

session_id = "Default Session"


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "Assistant", "content": "Ask your question, and I'll provide precise insights..."}]

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader(label='📂 Upload the documents (PDF files)', type='pdf', accept_multiple_files=True)

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]


# Whenever a document is uploaded it is split and embedded
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, 'wb') as file:
            file.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()
        documents.extend(docs)
    
    # Creating Splitter and Vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500)
    splits = text_splitter.split_documents(documents)
    vectordb = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectordb.as_retriever()


    prompt_template = """
    Given a chat history and the latest user query, rewrite the query into a standalone question that is fully self-contained and does not rely on prior context. Do not answer the question—only rephrase or return it as is if no changes are needed.
    """

    prompt = ChatPromptTemplate(
        [
            ('system', prompt_template),
            MessagesPlaceholder('chat_history'),
            ('user', '{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)


    sys_prompt = (
    """You are a highly intelligent AI financial analyst wit expertise in extracting key financial details from documents. Your responses are precise, structured, and professional, ensuring accuracy in financial data retrieval.
    \n\n
    Use the provided {context} to extract the following key details:

Company Name : The official name of the company mentioned in the document.
Report Date : The date the financial report was issued or published.
Profit Before Tax (PBT) : The pre-tax earnings of the company. If multiple values are present, extract the most relevant one based on context.
(Bonus) Additional Financial Details : Extract other key financial figures like revenue, net profit, operating expenses, and any notable financial indicators if available.
"""
    )

    q_a_prompt = ChatPromptTemplate(
        [
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ('user', '{input}'),
        ]
    )


    question_ans_chain = create_stuff_documents_chain(llm, q_a_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)


    conversational_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer',
    )
    response = conversational_chain.invoke(
            {'input': 'Extract financial details from the report.'},
            config={'configurable': {'session_id': session_id}}
        )
    
    st.write(response['answer'])




