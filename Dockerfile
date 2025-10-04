# version of base image
FROM python:3.11.13-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update


# Copy the rest of the application code
COPY ./data /app/data
COPY ./app.py /app/app.py
COPY ./models /app/models

EXPOSE 8080

# CMD provides the default arguments, which can be overridden
CMD [ "streamlit","run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0" ]