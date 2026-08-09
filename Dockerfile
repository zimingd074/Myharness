FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /tmp/rocketmq-client.deb \
      https://github.com/apache/rocketmq-client-cpp/releases/download/2.0.0/rocketmq-client-cpp-2.0.0.amd64.deb \
    && apt-get install -y /tmp/rocketmq-client.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/rocketmq-client.deb \
    && pip install --no-cache-dir -r requirements.txt
COPY evoagent ./evoagent
COPY web ./web
COPY skills ./skills
EXPOSE 8080
CMD ["python", "-m", "evoagent"]
