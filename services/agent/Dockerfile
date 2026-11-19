# Build from the repo root (needs services/rag/data/dog_breeds.csv):
#   docker build -f services/agent/Dockerfile -t dog-agent .
FROM python:3.13-slim
WORKDIR /srv
COPY services/agent/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt
COPY services/rag/data/dog_breeds.csv /srv/data/dog_breeds.csv
COPY services/agent/app /srv/app
ENV BREEDS_CSV=/srv/data/dog_breeds.csv
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
