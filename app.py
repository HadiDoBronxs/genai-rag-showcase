import streamlit as st
import os
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

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
    
    # Button zum Erzwingen eines Neuladens (falls neue PDFs hinzugefügt wurden)
    with st.sidebar:
        if st.button("🔄 Index aktualisieren"):
            st.session_state.knowledge_base = None
            # Vorhandenen Index-Ordner löschen oder ignorieren lassen
            # Wir setzen ein Flag im Session State
            st.session_state["force_refresh"] = True
            st.rerun()

    if st.session_state.knowledge_base is None:
        
        # Pfad zum gespeicherten Index
        index_folder = "faiss_index"
        
        # Check: Laden vom Disk möglich? (Nur wenn User nicht "Refresh" gedrückt hat)
        if os.path.exists(index_folder) and not st.session_state.get("force_refresh", False):
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                st.session_state.knowledge_base = FAISS.load_local(
                    index_folder, 
                    embeddings, 
                    allow_dangerous_deserialization=True # Vertrauenswürdiger lokaler Index
                )
                st.success("Wissensdatenbank vom Disk geladen! 🚀 (Schnellstart)")
            except Exception as e:
                st.error(f"Fehler beim Laden des Index: {e}")
                st.session_state["force_refresh"] = True # Fallback: Neu bauen
                st.rerun()
        
        # Fallback oder Refresh gewünscht: Neu erstellen
        else:
            documents = []
            data_folder = "data" # Der Ordner neben der app.py
            
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
                                for i, page in enumerate(pdf_reader.pages):
                                    page_text = page.extract_text()
                                    if page_text:
                                        # Metadaten für Zitate (Dateiname + Seitenzahl)
                                        # Verwende Document Objekt für LangChain
                                        from langchain_core.documents import Document
                                        doc = Document(
                                            page_content=page_text,
                                            metadata={"source": file, "page": i+1}
                                        )
                                        documents.append(doc)
                            except Exception as e:
                                st.error(f"Konnte {file} nicht lesen: {e}")
                                
                        status.update(label=f"Fertig! {len(files)} Dateien verarbeitet.", state="complete", expanded=False)
                    
                    # Wenn Dokumente gefunden wurden -> Vektordatenbank bauen
                    if documents:
                        # Upgrade: RecursiveCharacterTextSplitter für besseren Kontext
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        # split_documents behält Metadaten bei!
                        chunks = text_splitter.split_documents(documents)
                        
                        # Embeddings erstellen
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        st.session_state.knowledge_base = FAISS.from_documents(chunks, embeddings)
                        
                        # Index auf Disk speichern für schnelleren Start beim nächsten Mal
                        st.session_state.knowledge_base.save_local(index_folder)
                        
                        # Force Refresh Flag zurücksetzen
                        if "force_refresh" in st.session_state:
                            del st.session_state["force_refresh"]
                            
                        st.toast("Wissensdatenbank neu erstellt und gespeichert! 💾")
                else:
                    st.warning(f"Der Ordner '{data_folder}' ist leer. Bitte lege PDFs hinein.")
            else:
                st.error(f"Der Ordner '{data_folder}' wurde nicht gefunden.")
                # Fallback: Manueller Upload Button
                pdf = st.file_uploader("Alternative: Lade hier eine PDF hoch", type="pdf")
                if pdf:
                    from langchain_core.documents import Document
                    pdf_reader = PdfReader(pdf)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            doc = Document(
                                page_content=text,
                                metadata={"source": pdf.name, "page": i+1}
                            )
                            documents.append(doc)
                    if documents:
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(documents)
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        st.session_state.knowledge_base = FAISS.from_documents(chunks, embeddings)

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
                # Callback Handler für Streaming (Schreibmaschinen-Effekt)
                from langchain_community.callbacks import StreamlitCallbackHandler
                st_callback = StreamlitCallbackHandler(st.container())
                
                with st.spinner("Suche Antworten..."):
                    # 1. Relevante Textstellen finden
                    docs = st.session_state.knowledge_base.similarity_search(prompt)
                    
                    # 2. GPT-3.5 fragen (mit Streaming aktiviert)
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, streaming=True)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # System Prompt aktivieren
                    system_instruction = "Du bist ein professioneller Assistent für Bewerbungsunterlagen. Antworte basierend auf dem Kontext. Wenn du etwas nicht weißt, sage es."
                    
                    # Callback übergeben, damit der Text live erscheint
                    response = chain.run(
                        input_documents=docs, 
                        question=f"{system_instruction}\nFrage: {prompt}",
                        callbacks=[st_callback]
                    )
                    
                    # Falls der Callback das Markdown nicht final rendert (manchmal nötig):
                    # st.markdown(response) -> Der Callback macht das meist schon, aber sicherheitshalber lassen wir es weg, um Doppelung zu vermeiden.
                    # Wenn der Callback fertig ist, steht der Text da.
                    
                    # 3. Quellen anzeigen
                    with st.expander("📚 Verwendete Quellen anzeigen"):
                        seen_sources = set()
                        for doc in docs:
                            source_id = f"{doc.metadata.get('source', 'Unbekannt')} (Seite {doc.metadata.get('page', '?')})"
                            if source_id not in seen_sources:
                                st.write(f"- {source_id}")
                                # Optional: Vorschau des Textes
                                # st.caption(doc.page_content[:150] + "...")
                                seen_sources.add(source_id)
                    
            # Antwort und Quellen in History speichern (vereinfacht speichern wir nur den Text)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()