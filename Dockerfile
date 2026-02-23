FROM python:3.14-slim

WORKDIR /app

# Optional faster/cleaner Python output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Install dependencies directly into the system Python in the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY crawler.py ./

# Create empty output directory (will be populated by the crawler)
RUN mkdir -p /app/out

# Execute the crawler directly on start
CMD ["python", "crawler.py"]