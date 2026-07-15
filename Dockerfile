FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/localcode
COPY pyproject.toml README.md ./
COPY localcode ./localcode
RUN pip install --no-cache-dir ".[treesitter,api]"

# non-root user; workspace is mounted at /workspace
RUN useradd -m coder
USER coder
WORKDIR /workspace

# point at the host's or a linked Ollama by default
ENV LOCALCODE_BASE_URL=http://ollama:11434

ENTRYPOINT ["lc"]
CMD ["/workspace"]
