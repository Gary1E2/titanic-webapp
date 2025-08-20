FROM python:3.10-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Shared dir is mounted by compose; create just in case
RUN mkdir -p /usr/src/app/shared_volume
EXPOSE 5000
CMD ["python", "app.py"]
