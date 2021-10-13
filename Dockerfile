# syntax=docker/dockerfile:1
FROM python:3.8

RUN pip install flask

COPY . .

CMD ["python", "log_display_app.py"]

 