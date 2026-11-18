FROM python:3.12-slim
COPY index.html /app/index.html
COPY receipt_exhibit.mjs /app/receipt_exhibit.mjs
COPY exhibit_app.mjs /app/exhibit_app.mjs
COPY favicon.svg /app/favicon.svg
WORKDIR /app
EXPOSE 7860
CMD ["python", "-m", "http.server", "7860", "--bind", "0.0.0.0"]
