FROM python:3.12-slim
COPY index.html /app/index.html
WORKDIR /app
EXPOSE 7860
CMD ["python", "-m", "http.server", "7860", "--bind", "0.0.0.0"]
