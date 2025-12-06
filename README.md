# 🤖 GenAI RAG Showcase: Enterprise Document Assistant

Eine **End-to-End Retrieval Augmented Generation (RAG)** Anwendung, die unstrukturierte Daten mittels Vektorisierung und LLMs durchsuchbar macht.

---

### 🚀 Live Demo: Chat with my Portfolio
Diese Instanz der Applikation wurde speziell für Bewerbungszwecke deployt. Sie ist mit meinen **persönlichen Unterlagen** (Lebenslauf, Arbeitszeugnis Bosch, Zertifikate) "gefüttert".

**Probieren Sie es aus! Stellen Sie Fragen wie:**
* *"Welche Technologien hat Hadi bei Bosch eingesetzt?"*
* *"Fasse seinen Bildungsweg zusammen."*

---

### 🎯 Abstract & Business Use Case
Technisch demonstriert dieses Projekt, wie Unternehmen internes Wissen (z.B. Handbücher, technische Doku) effizient zugänglich machen können. Anstatt Dokumente manuell zu durchsuchen, ermöglicht die RAG-Architektur eine intelligente "Chat with your Data"-Schnittstelle.

**Kernfunktionen:**
* **Multi-Document Ingestion:** Automatisches Einlesen ganzer Dokumenten-Ordner.
* **Semantic Search:** Finden von Inhalten anhand der *Bedeutung* (Vektorsuche via FAISS).
* **Context Awareness:** Die KI nutzt nur die bereitgestellten Fakten für Antworten (Vermeidung von Halluzinationen).

### ⚙️ Technische Architektur
Der Workflow folgt dem modernen RAG-Pattern:

1.  **Ingestion:** Parsing von PDF-Dokumenten (`pypdf`).
2.  **Chunking:** Aufteilung von Text in semantische Abschnitte (`RecursiveCharacterTextSplitter`).
3.  **Embedding:** Umwandlung von Text in Vektoren (`OpenAI Ada-002`).
4.  **Vector Store:** Speicherung in einer lokalen Vektordatenbank (`FAISS`) für O(1) Retrieval-Performance.
5.  **Generation:** Kontext-basierte Antwortgenerierung durch `GPT-3.5-Turbo` via `LangChain`.

### 🛠 Tech Stack

| Komponente | Technologie |
| :--- | :--- |
| **Sprache** | Python 3.9+ |
| **Orchestrierung** | LangChain |
| **Frontend** | Streamlit |
| **LLM Provider** | OpenAI API |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |

### 🔒 Security & Privacy
* **API Key Management:** Der OpenAI Key wird sicher über `st.secrets` (Cloud) verwaltet.
* **Datenschutz:** Dokumente werden lokal verarbeitet. In dieser Demo-Version sind meine Unterlagen serverseitig hinterlegt, sodass kein Upload notwendig ist.

---
*Developed by Hadi Nasrullah | Focus: Software Engineering & AI Integration*
