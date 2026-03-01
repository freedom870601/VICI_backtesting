FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port", "8080", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false", \
     "--browser.gatherUsageStats", "false"]
