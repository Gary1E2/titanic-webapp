FROM python:3.10-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /usr/src/app/shared_volume
EXPOSE 8000
CMD ["python", "processing_api.py"]
