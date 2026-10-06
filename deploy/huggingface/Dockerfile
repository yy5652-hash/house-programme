# Hugging Face Space for House Programme.
# The app's source is https://github.com/yy5652-hash/house-programme ; APP_COMMIT pins the commit that is deployed.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
ARG APP_COMMIT=5818ec9868d6edd03d006a5040c2b101e9a83de3
RUN pip install --no-cache-dir "taste-mcp[app] @ https://github.com/yy5652-hash/house-programme/archive/${APP_COMMIT}.zip"
RUN useradd -m -u 1000 app
USER app
EXPOSE 7860
CMD ["house-programme"]
