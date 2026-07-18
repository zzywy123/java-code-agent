FROM python:3.11-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels ".[ui]"

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/agent \
    HF_HOME=/home/agent/.cache/huggingface

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        git \
        openjdk-17-jdk-headless \
        maven \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system agent \
    && useradd --system --gid agent --home-dir /home/agent --create-home agent \
    && mkdir -p /home/agent/workspace /home/agent/data/index \
        /home/agent/data/checkpoints /home/agent/data/memory \
        /home/agent/data/observability /home/agent/data/workspaces \
        /home/agent/.cache/huggingface \
        /home/agent/.m2 /opt/agent \
    && chown -R agent:agent /home/agent /opt/agent

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY --chown=agent:agent src/agent/ui/app.py /opt/agent/app.py

USER agent
WORKDIR /home/agent/workspace
EXPOSE 8501

CMD ["streamlit", "run", "/opt/agent/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
