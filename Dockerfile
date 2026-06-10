FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata && rm -rf /var/lib/apt/lists/*
ENV TZ=Europe/Paris

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY config.yaml ./
COPY wvs7.json ./
COPY persona_prompts.json ./

RUN mkdir -p /app/results /app/logs

# Daily run at 11:00 CET (= 10:00 UTC; TZ=Europe/Paris handles DST)
RUN echo "0 11 * * * cd /app && /usr/local/bin/python3 pipeline.py run >> /app/logs/cron.log 2>&1" | crontab -

CMD ["sh", "-c", "printenv > /etc/environment && cron -f"]
