FROM python:3.13
WORKDIR /srv
COPY ./requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /srv/requirements.txt
COPY ./app /srv/app
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]
