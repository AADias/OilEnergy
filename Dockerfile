FROM python:3.12-slim

WORKDIR /app

# Install optional dependencies (enhanced models, charts)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-fetch weather/demand context data using synthetic fallback
RUN python scripts/fetch_weather.py --synthetic

# Expose web server port
EXPOSE 8080

# Run the web server
CMD ["python", "web_server.py", "--host", "0.0.0.0", "--port", "8080"]
