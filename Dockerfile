# Use an official Python slim image
FROM python:3.12.4-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# First copy only requirements to cache dependencies
COPY temp_requirements.txt .

# Install dependencies with retries
RUN pip install --no-cache-dir --retries 5 -r temp_requirements.txt

# Now copy the rest of the application
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
