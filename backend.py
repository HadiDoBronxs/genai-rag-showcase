import os
import streamlit as st
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource(show_spinner=False)
def load_knowledge_base(openai_api_key):
    """
    Lädt die Wissensdatenbank.
    Diese Funktion wird gecacht, damit sie nicht bei jedem Rerun neu ausgeführt wird.
    """
    # 1. Versuch: Lokal gespeicherten Index laden
    index_folder = "faiss_index"
    if os.path.exists(index_folder):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
            print("Index vom Disk geladen.") 
            return db
        except Exception as e:
            print(f"Fehler beim Laden des Index: {e}")
            pass

    # 2. Versuch: Neu erstellen
    documents = []
    data_folder = "data"
    
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
        
        if files:
            for file in files:
                pdf_path = os.path.join(data_folder, file)
                try:
                    pdf_reader = PdfReader(pdf_path)
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            doc = Document(
                                page_content=page_text,
                                metadata={"source": file, "page": i+1}
                            )
                            documents.append(doc)
                except Exception as e:
                    print(f"Fehler beim Lesen von {file}: {e}")
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                db = FAISS.from_documents(chunks, embeddings)
                
                # Speichern
                db.save_local(index_folder)
                return db
    
    return None
