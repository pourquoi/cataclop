FROM python:3.7
ENV PYTHONUNBUFFERED=1

RUN pip install poetry
RUN pip install supervisor

COPY .env /app/.env
COPY pyproject.toml /app/pyproject.toml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app

RUN poetry install
RUN export $(grep -v '^#' .env | xargs -0)

COPY . /app

ENTRYPOINT ["/entrypoint.sh"]