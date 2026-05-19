# Path-landscape webapp — production container
# Used by Render / Fly / Railway / any container host.
#
# Build:   docker build -t path-landscape .
# Run:     docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... path-landscape
FROM python:3.12-slim

# System deps for matplotlib (libgomp), networkx, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
        libfreetype6 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the source
COPY path_landscape ./path_landscape
COPY serve_agent.py .

# Where per-job artifacts (figures, JSON) get written
ENV PATH_LANDSCAPE_OUT=/data/runs
RUN mkdir -p /data/runs
VOLUME ["/data/runs"]

# Render/Fly/Railway set $PORT; default to 8080 for local docker
ENV PORT=8080
EXPOSE 8080

# Gunicorn is the production WSGI server. The Flask app instance is
# created on import of path_landscape.webapp.server.
CMD exec gunicorn \
        --bind "0.0.0.0:${PORT}" \
        --workers 1 \
        --threads 4 \
        --timeout 300 \
        --access-logfile - \
        "path_landscape.webapp.server:app"
