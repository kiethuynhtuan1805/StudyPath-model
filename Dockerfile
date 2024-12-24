FROM python:3.11.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y openjdk-17-jdk procps curl && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
