FROM python:3.10-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
RUN .venv/bin/pip install gunicorn

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
