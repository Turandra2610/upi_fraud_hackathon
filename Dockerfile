FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install flask gymnasium numpy torch
# This is the line the bot is looking for:
CMD ["python", "server/app.py"]
