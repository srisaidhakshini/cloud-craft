FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends iproute2 gosu \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY cloudcraft ./cloudcraft
COPY configs ./configs
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "cloudcraft.client"]
