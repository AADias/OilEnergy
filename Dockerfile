FROM python:3.12-slim

WORKDIR /app

# Copy project (no optional heavyweight dependencies required for core CLI)
COPY . .

# Set PYTHONPATH so the src layout works
ENV PYTHONPATH=/app/src

# Context data (weather.csv, demand.csv) must be provided by the user at
# runtime via a volume mount or by running fetch_weather.py before building.
# Do NOT embed synthetic/demo data in the image.
#
# To provide real context data before building:
#   python scripts/fetch_weather.py
#   docker build -t oilenergy .
#
# Or mount at runtime:
#   docker run -v /path/to/data/context:/app/data/context oilenergy
#
# To fetch context data inside the container after start:
#   docker run -it oilenergy python scripts/fetch_weather.py

# Run the CLI by default; override CMD to use other entry points
CMD ["python", "scripts/train_model.py"]
