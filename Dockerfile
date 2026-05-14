FROM python:3.10-slim

# -------------------------------------------------
# Environment variables
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------
# Set working directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# Install system dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Copy requirements first (Docker cache optimization)
# -------------------------------------------------
COPY flask_app/requirements.txt .

# -------------------------------------------------
# Install Python dependencies
# -------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Download NLTK data
# -------------------------------------------------
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

# -------------------------------------------------
# Copy application code
# -------------------------------------------------
COPY flask_app/ /app/

# -------------------------------------------------
# Copy model artifacts
# -------------------------------------------------
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# -------------------------------------------------
# Expose Flask/Gunicorn port
# -------------------------------------------------
EXPOSE 5000

# -------------------------------------------------
# Production server
# -------------------------------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
