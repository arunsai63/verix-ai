FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libmagic1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.4

# Configure Poetry to not create virtual environments since we're in Docker
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies using Poetry
RUN poetry install --no-dev --no-root

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/datasets /app/chroma_db

# Install the project (in case it's a package)
RUN poetry install --no-dev

# Expose port
EXPOSE 8000

# Run the application  
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]