FROM tensorflow/tensorflow:2.5.0

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir Flask gunicorn

EXPOSE 6547
CMD ["gunicorn", "-b", "0.0.0.0:6547", "app:app"]
