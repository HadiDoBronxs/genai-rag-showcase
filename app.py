import streamlit as st
import os
import shutil
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

# Custom Handler für Streaming (Global definiert)
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
            # Hinweis: Wir zeigen success innerhalb der Funktion, 
            # aber da sie gecacht ist, sieht man es nur beim ersten Laden.
            # Das ist okay oder wir nutzen st.toast.
            print("Index vom Disk geladen.") 
            return db
        except Exception as e:
            print(f"Fehler beim Laden des Index: {e}")
            # Bei Fehler Fallback auf Neu-Erstellung
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

def main():
    st.set_page_config(page_title="Chat with Hadi's Docs 📂", page_icon="🤖")
    
    # --- 1. SESSION STATE INITIALISIEREN ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 2. API KEY MANAGEMENT ---
    api_key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass

    if not api_key:
        with st.sidebar:
            st.title("⚙️ Einstellungen")
            api_key = st.text_input("OpenAI API Key:", type="password")
            st.markdown("---")
            st.caption("Der Key wird nur für diese Sitzung genutzt.")
    
    if not api_key:
        st.info("👋 Willkommen! Bitte gib links deinen OpenAI API Key ein, um zu starten.")
        st.stop()

    st.header("🤖 Chat with Hadi's Portfolio")
    st.caption("Frage mich etwas zu Lebenslauf, Zeugnissen oder Zertifikaten!")

    # --- 3. WISSENSDATENBANK LADEN ---
    
    # Refresh Button
    with st.sidebar:
        if st.button("🔄 Index aktualisieren"):
            # 1. Folder löschen
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            # 2. Cache löschen
            load_knowledge_base.clear()
            # 3. Rerun
            st.rerun()

    # Laden mit Cache
    if "knowledge_base" not in st.session_state:
         # Optional: wir nutzen direkt den Rückgabewert, aber für Kompatibilität
         # mit altem Code können wir es zuweisen, oder den Code unten anpassen.
         # Um den Refactor minimal-invasiv zu halten beim Verwenden unten:
         pass

    # Wir rufen die Funktion auf. Dank Cache ist das billig.
    # Wir zeigen einen Spinner nur wenn es wirklich lädt (da show_spinner=False im Decorator, machen wir es hier manuell wenn wir wollen, oder verlassen uns auf Speed)
    # Da Cache beim ersten Mal dauert, wäre ein Spinner gut.
    # Aber wir können nicht wissen ob es gecacht ist ohne Hacks.
    # Einfach aufrufen:
    with st.spinner("Lade Wissensdatenbank..."):
        knowledge_base = load_knowledge_base(api_key)
    
    if knowledge_base is None:
        st.warning("Keine Dokumente gefunden oder Fehler beim Erstellen der Datenbank.")
        st.stop()
    
    # Session State Update für Kompatibilität (falls nötig, aber eigentlich nutzen wir local var)
    st.session_state.knowledge_base = knowledge_base

    # --- 4. CHAT INTERFACE ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Deine Frage an die Unterlagen..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            st_callback = StreamHandler(response_placeholder)
            
            # Relevante Textstellen finden
            docs = knowledge_base.similarity_search(prompt)
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, streaming=True)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            system_instruction = "Du bist ein enthusiastischer HR-Assistent. Hebe Hadis Stärken besonders hervor und antworte professionell aber überzeugend. Antworte basierend auf dem Kontext. Wenn du etwas nicht weißt, sage es."
            
            response = chain.run(
                input_documents=docs, 
                question=f"{system_instruction}\nFrage: {prompt}",
                callbacks=[st_callback]
            )
            
            # Quellen anzeigen
            with st.expander("📚 Verwendete Quellen anzeigen"):
                seen_sources = set()
                for doc in docs:
                    source_id = f"{doc.metadata.get('source', 'Unbekannt')} (Seite {doc.metadata.get('page', '?')})"
                    if source_id not in seen_sources:
                        st.write(f"- {source_id}")
                        seen_sources.add(source_id)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()