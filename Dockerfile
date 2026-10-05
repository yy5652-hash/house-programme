FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]"
RUN useradd -m -u 1000 app
USER app
EXPOSE 7860
CMD ["house-programme"]
