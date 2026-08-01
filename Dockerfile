# StockAI API container
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source plus the trained weights and their preprocessing specs. See
# .dockerignore for what is deliberately left out of the image.
COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
