import streamlit as st
import os
import shutil
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from backend import StreamHandler, load_knowledge_base

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

    with st.spinner("Lade Wissensdatenbank..."):
        knowledge_base = load_knowledge_base(api_key)
    
    if knowledge_base is None:
        st.warning("Keine Dokumente gefunden oder Fehler beim Erstellen der Datenbank.")
        st.stop()
    
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
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, streaming=True, temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            system_instruction = "Du bist ein professioneller HR-Assistent. Antworte NUR basierend auf dem folgenden Kontext. Wenn die Antwort nicht im Kontext enthalten ist, sage explizit 'Ich weiß es nicht' und erfinde keine Fakten. Hebe Hadis Stärken hervor, wo es der Kontext belegt."
            
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