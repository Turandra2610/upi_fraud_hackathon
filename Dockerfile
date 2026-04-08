FROM ghcr.io/meta-pytorch/openenv-base:latest

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . .

RUN pip install --no-cache-dir numpy

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
