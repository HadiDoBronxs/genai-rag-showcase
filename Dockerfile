# Wir nutzen ein leichtes Python-Image als Basis
FROM python:3.10-slim

# Arbeitsverzeichnis im Container setzen
WORKDIR /app

# System-Abhängigkeiten installieren (wichtig für manche Python-Pakete)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiere die Requirements zuerst (für besseres Caching)
COPY requirements.txt .

# Installiere die Python-Bibliotheken
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Codes in den Container
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Port für Streamlit freigeben
EXPOSE 8501

# Healthcheck (Optional, aber professionell für Cloud-Umgebungen)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Der Startbefehl
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]