FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y 
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python", "app.py"]