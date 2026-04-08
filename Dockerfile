FROM python:3.9

# Standard Hugging Face user setup
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy ALL files (including the new server folder) into the container
COPY --chown=user . .

# Install your libraries
RUN pip install --no-cache-dir flask gymnasium numpy torch

# This MUST match the folder name you created
CMD ["python", "server/app.py"]
