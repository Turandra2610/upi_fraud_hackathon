FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install flask gymnasium numpy torch
CMD ["python", "app.py"]
