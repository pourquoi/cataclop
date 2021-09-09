FROM python:3.7
ENV PYTHONUNBUFFERED=1

RUN pip install poetry
RUN pip install supervisor

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN poetry install

COPY .env /app/.env
RUN export $(grep -v '^#' .env | xargs -0)

COPY . /app

COPY docker/uwsgi.ini /app/uwsgi.ini

COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]