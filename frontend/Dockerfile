FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (for PDF processing, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy and install frontend requirements
COPY frontend/requirements.txt ./frontend/
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy model-training scripts (needed by QueryHandler)
# These are imported by frontend/query_handler.py
COPY model-training/scripts/ ./model-training/scripts/

# Copy shared utilities if they exist
COPY shared/ ./shared/ 2>/dev/null || true

# Copy data-pipeline scripts if needed for unstructured data
COPY data-pipeline/scripts/ ./data-pipeline/scripts/ 2>/dev/null || true

# Copy frontend application
COPY frontend/ ./frontend/

# Set Python path to include model-training scripts and shared utilities
ENV PYTHONPATH="/app:/app/model-training/scripts:/app/shared:/app/data-pipeline/scripts:${PYTHONPATH}"

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]