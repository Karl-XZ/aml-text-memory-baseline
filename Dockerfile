FROM python:3.12-slim

WORKDIR /app
COPY server.py /app/server.py
RUN mkdir -p /app/state
EXPOSE 8000
VOLUME ["/app/state"]
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--db", "/app/state/memory.sqlite3"]
