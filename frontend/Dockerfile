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

# Create directories for optional dependencies
RUN mkdir -p ./shared ./data-pipeline/scripts

# Copy helper script
COPY frontend/copy-optional.sh /tmp/copy-optional.sh
RUN chmod +x /tmp/copy-optional.sh

# Copy entire build context to temp location for optional directory handling
# This allows us to check if directories exist before copying
COPY . /tmp/build-context/

# Run helper script to conditionally copy from temp location to final location
RUN /tmp/copy-optional.sh && rm -rf /tmp/build-context /tmp/copy-optional.sh || true

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