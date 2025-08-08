import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory


load_dotenv()

# --- Caching Functions for Performance ---
@st.cache_resource(show_spinner="Processing documents...")
def setup_vectorstore(uploaded_files):
    all_splits = []
    for uploaded_file in uploaded_files:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)
    
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings()
    )
    return vectorstore

def create_agent_executor(vectorstore, selected_docs):
    
    #THis Fuction Creates the agent, its tools, and the executor that runs it.
   
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    
    full_selected_doc_paths = [os.path.join(tempfile.gettempdir(), doc_name) for doc_name in selected_docs]

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": {"source": {"$in": full_selected_doc_paths}} if full_selected_doc_paths else None
        }
    )
    retriever_tool = create_retriever_tool(
        retriever,
        "document_qa_retriever",
        "Use this tool to answer questions using the content of the uploaded PDF documents."
    )

    # Prompt for the agent
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized Q&A and summarization assistant. Your purpose is to answer questions or provide summaries based ONLY on the context provided by the document retriever tool. "
         "If the user asks for a summary, provide a concise summary of the provided context. "
         "If the user asks a question and the context contains the answer, provide it and cite the source document and page number. "
         "If the context does not contain enough information to answer a question, you MUST reply with the exact phrase: 'The answer is not found in the provided documents.' "
         "Do not use any external knowledge. Do not engage in any conversation. Do not explain why you cannot answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


    agent = create_openai_functions_agent(llm, [retriever_tool], agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retriever_tool],
        verbose=True,
        memory=st.session_state.memory
    )
    return agent_executor


# --- STREAMLIT UI ----
st.set_page_config(page_title="🤖 Production RAG Agent", layout="wide")
st.title("🤖 RAG Chatbot")
st.markdown("Upload PDFs, filter by document, and ask the agent anything!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        st.session_state.vectorstore = setup_vectorstore(uploaded_files)
        st.session_state.doc_names = [f.name for f in uploaded_files]
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.success(f"Processed {len(uploaded_files)} documents!")

 

if not st.session_state.vectorstore:
    st.info("Please upload and process your documents in the sidebar to start the chat.")
else:
    if not st.session_state.agent_executor:
        st.session_state.agent_executor = create_agent_executor(st.session_state.vectorstore, [])
        st.info("Agent is ready. Ask a question!")
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                response = st.session_state.agent_executor.invoke({"input": user_query})
                st.markdown(response["output"])
        
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})