"""
Support file upload for analysis with conversation history tracking
"""
import time
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models import ChatZhipuAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
import requests

# Configuration
DEFAULT_DEVICE = "cuda"
DEFAULT_MEMORY = 6
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_JSON_CHUNK_SIZE = 300

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, device=DEFAULT_DEVICE, embedding_model_path=None):
        self.device = device
        self.embedding_model_path = embedding_model_path or os.getenv("EMBEDDING_MODEL_PATH")
        
    def load_document(self, file_path):
        """Load document based on file extension"""
        try:
            file_type = os.path.splitext(file_path)[1][1:].lower()
            
            loaders = {
                "txt": lambda: TextLoader(file_path, encoding="utf-8"),
                "pdf": lambda: PyPDFLoader(file_path),
                "csv": lambda: CSVLoader(file_path, encoding="utf-8"),
            }
            
            if file_type == "json":
                json_data = requests.get(file_path).json()
                splitter = RecursiveJsonSplitter(max_chunk_size=DEFAULT_JSON_CHUNK_SIZE)
                return splitter.create_documents(texts=[json_data])
            
            loader = loaders.get(file_type)
            if loader:
                return loader().load()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            st.error(f"Error loading document {file_path}: {e}")
            return None

    def process_and_store_document(self, file_path, persist_directory="chromaDB"):
        """Process document and store in vector database"""
        try:
            # Load document
            documents = self.load_document(file_path)
            if not documents:
                return False

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE, 
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            # Create embeddings and store
            if not self.embedding_model_path:
                st.error("Embedding model path not configured")
                return False
                
            embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path,
                model_kwargs={"device": self.device}
            )
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=persist_directory
            )
            
            st.success(f"Successfully processed: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return False

class LCAChat:
    """Main chat interface for LCA-GPT"""
    
    def __init__(self, memory_limit=DEFAULT_MEMORY):
        self.memory_limit = memory_limit
        self.doc_processor = DocumentProcessor()
        
    def setup_llm(self):
        """Initialize the language model"""
        load_dotenv()
        
        # Check for API keys in environment variables
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_base = os.getenv("OPENAI_API_BASE")
        
        if dashscope_key:
            return ChatTongyi(streaming=True, model='qwen-max')
        elif openai_key:
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                openai_api_key=openai_key,
                openai_api_base=openai_base
            )
        else:
            st.error("No API keys found. Please configure DASHSCOPE_API_KEY or OPENAI_API_KEY in your .env file")
            return None

    def create_rag_chain(self, persist_directory="chromaDB"):
        """Create the RAG chain"""
        try:
            llm = self.setup_llm()
            if not llm:
                return None
                
            # Setup embeddings and retriever
            embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
            if not embedding_model_path:
                st.error("EMBEDDING_MODEL_PATH not configured in environment")
                return None
                
            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": self.doc_processor.device}
            )
            
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
            )
            retriever = vectorstore.as_retriever()

            # Create prompt template
            system_prompt = (
                "You are an experienced expert in the Life Cycle Assessment (LCA) field. "
                "Use the retrieved context to answer questions. If the context doesn't have "
                "sufficient information, please indicate this clearly. "
                "If you have any uncertainties, please ask the user for clarification. "
                "For questions related to LCA evaluation, you should provide complete citations "
                "to reference materials.\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # Create chains
            qa_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            
            return rag_chain
            
        except Exception as e:
            st.error(f"Error creating RAG chain: {e}")
            return None

    def process_user_input(self, user_input, rag_chain, chat_history):
        """Process user input and generate response"""
        try:
            if not rag_chain:
                return "System not properly initialized. Please check configuration."
                
            # Limit chat history to prevent memory overflow
            limited_history = chat_history[-self.memory_limit:] if len(chat_history) > self.memory_limit else chat_history
            
            result = rag_chain.invoke({
                "input": user_input,
                "chat_history": limited_history
            })
            
            return result.get("answer", "No response generated")
            
        except Exception as e:
            st.error(f"Error processing input: {e}")
            return "An error occurred while processing your request."

@st.cache_resource(ttl="1h")
def process_uploaded_files(uploaded_files):
    """Process uploaded files and store in vector database"""
    if not uploaded_files:
        return
        
    doc_processor = DocumentProcessor()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
                
            doc_processor.process_and_store_document(temp_filepath)

def main():
    """Main Streamlit application"""
    start_time = time.time()
    
    st.set_page_config(page_title="LCA-GPT", layout="wide")
    st.title("LCA-GPT - Life Cycle Assessment Assistant")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            label="Upload files for analysis",
            type=["txt", "pdf", "csv", "json"],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, CSV, JSON"
        )
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                process_uploaded_files(uploaded_files)
        
        if st.button("Clear Chat History"):
            st.session_state.clear()
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm LCA-GPT, your Life Cycle Assessment assistant. Upload documents and ask me questions about LCA methodology, analysis, or interpretation."}
        ]
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "lca_chat" not in st.session_state:
        st.session_state.lca_chat = LCAChat()
    
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.rag_chain = st.session_state.lca_chat.create_rag_chain()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your LCA-related question:"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.lca_chat.process_user_input(
                    prompt,
                    st.session_state.rag_chain,
                    st.session_state.chat_history
                )
            
            st.markdown(response)

        # Update session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])

    # Display performance info
    execution_time = time.time() - start_time
    st.sidebar.caption(f"Page load time: {execution_time:.2f}s")

if __name__ == "__main__":
    main()