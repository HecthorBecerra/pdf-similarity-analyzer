# -- Imagen base ─────────────────────────────────────────────────────
FROM python:3.11-slim

# -- Variables de entorno ────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# -- Directorio de trabajo ──────────────────────────────────────────
WORKDIR /app

# -- Dependencias del sistema ───────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# -- Dependencias de Python (capa cacheada) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- Codigo fuente de la aplicacion ─────────────────────────────────
COPY . .

# -- Exponer puerto de Streamlit ────────────────────────────────────
EXPOSE 8501

# -- Ejecutar la aplicacion Streamlit ───────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

# -- Comandos rapidos de inicio ─────────────────────────────────────
# docker build -t pdf-similarity .
# docker run -p 8501:8501 pdf-similarity
# Luego abrir: http://localhost:8501
