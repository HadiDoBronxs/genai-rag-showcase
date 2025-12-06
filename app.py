import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import load_qa_chain

def main():
    st.set_page_config(page_title="Chat with Hadi's Docs 📂", page_icon="🤖")
    
    # --- 1. SESSION STATE INITIALISIEREN ---
    # Damit sich der Chatbot an den Verlauf erinnert
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Damit wir die PDFs nicht bei jeder Frage neu laden müssen
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # --- 2. API KEY MANAGEMENT (Absturzsicher!) ---
    api_key = None
    
    # Wir versuchen, den Key aus den Secrets zu laden (für Cloud Deployment)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        # Lokal gibt es oft keine secrets.toml -> Einfach ignorieren
        pass
    except Exception:
        pass

    # Wenn kein Key gefunden wurde (also lokal), fragen wir den User
    if not api_key:
        with st.sidebar:
            st.title("⚙️ Einstellungen")
            api_key = st.text_input("OpenAI API Key:", type="password")
            st.markdown("---")
            st.caption("Der Key wird nur für diese Sitzung genutzt.")
    
    # Harter Stopp, wenn immer noch kein Key da ist
    if not api_key:
        st.info("👋 Willkommen! Bitte gib links deinen OpenAI API Key ein, um zu starten.")
        st.stop()

    st.header("🤖 Chat with Hadi's Portfolio")
    st.caption("Frage mich etwas zu Lebenslauf, Zeugnissen oder Zertifikaten!")

    # --- 3. DOKUMENTE LADEN (Ordner 'data') ---
    if st.session_state.knowledge_base is None:
        text = ""
        data_folder = "data" # Der Ordner neben der app.py
        loaded_files = []
        
        # Prüfen, ob der Ordner existiert und Dateien enthält
        if os.path.exists(data_folder):
            files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
            
            if files:
                with st.status("Lade Dokumente aus 'data' Ordner...", expanded=True) as status:
                    for file in files:
                        st.write(f"📄 Lese {file}...")
                        pdf_path = os.path.join(data_folder, file)
                        
                        try:
                            pdf_reader = PdfReader(pdf_path)
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text
                            loaded_files.append(file)
                        except Exception as e:
                            st.error(f"Konnte {file} nicht lesen: {e}")
                            
                    status.update(label=f"Fertig! {len(loaded_files)} Dokumente verarbeitet.", state="complete", expanded=False)
                
                # Wenn Text gefunden wurde -> Vektordatenbank bauen
                if text:
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    
                    # Embeddings erstellen
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
                    st.toast("Wissensdatenbank ist bereit! 🧠")
            else:
                st.warning(f"Der Ordner '{data_folder}' ist leer. Bitte lege PDFs hinein.")
        else:
            st.error(f"Der Ordner '{data_folder}' wurde nicht gefunden.")
            # Fallback: Manueller Upload Button
            pdf = st.file_uploader("Alternative: Lade hier eine PDF hoch", type="pdf")
            if pdf:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                if text:
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                    chunks = text_splitter.split_text(text)
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)

    # --- 4. CHAT INTERFACE ---
    # Alten Verlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Neue Eingabe verarbeiten
    if prompt := st.chat_input("Deine Frage an die Unterlagen..."):
        
        # User Frage anzeigen
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # KI Antwort generieren
        if st.session_state.knowledge_base:
            with st.chat_message("assistant"):
                with st.spinner("Suche Antworten..."):
                    # 1. Relevante Textstellen finden
                    docs = st.session_state.knowledge_base.similarity_search(prompt)
                    
                    # 2. GPT-3.5 fragen
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Optional: System-Prompting (KI Verhalten steuern)
                    # response = chain.run(input_documents=docs, question=f"Du bist ein HR-Assistent. Beantworte basierend auf den Unterlagen: {prompt}")
                    response = chain.run(input_documents=docs, question=prompt)
                    
                    st.markdown(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()