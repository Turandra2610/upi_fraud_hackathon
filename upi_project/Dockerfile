FROM python:3.9-slim

WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies one by one to avoid index confusion
RUN pip install --upgrade pip
RUN pip install flask numpy gymnasium
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5000

CMD ["python", "server/app.py"]
