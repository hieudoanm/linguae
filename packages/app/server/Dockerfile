FROM python:3.14.8-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY packages/app/server/pyproject.toml packages/app/server/uv.lock ./packages/app/server/
COPY packages/app/server/src ./packages/app/server/src
COPY packages/models/artifacts ./packages/models/artifacts

WORKDIR /app/packages/app/server

RUN uv sync --frozen

ENV LINGUAE_SERVER_HOST=0.0.0.0
ENV LINGUAE_SERVER_PORT=8000
ENV LINGUAE_DEVICE=cpu

EXPOSE 8000

CMD ["uv", "run", "linguae-server"]
