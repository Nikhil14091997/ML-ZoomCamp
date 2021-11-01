FROM python:3.8.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "final_model=1.0.bin", "./"]

EXPOSE 9697

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9697", "predict:app"]