# Creating a python base with shared environment variables
FROM python:3.7.4-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
        libmariadbclient-dev

WORKDIR /app
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash
RUN apt-get install -y nodejs
RUN npm install --global yarn
COPY package.json yarn.* /app/
RUN yarn install --prod

ENV POETRY_VERSION=1.0.5
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./poetry.* ./
RUN poetry install --no-dev  # respects



FROM python-base as development

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
        libmariadbclient-dev

RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash
RUN apt-get install -y nodejs
RUN npm install --global yarn

COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

RUN pip install supervisor

# Copying in our entrypoint
COPY ./docker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# venv already has runtime deps installed we get a quicker install
WORKDIR $PYSETUP_PATH
RUN poetry install

WORKDIR /app
COPY . .

COPY .env /app/.env
RUN export $(grep -v '^#' .env | xargs -0)

COPY --from=builder-base /app/node_modules /app/node_modules
RUN yarn install

COPY docker/supervisor/supervisord.conf /etc/supervisor/supervisord.conf
COPY docker/uwsgi.ini /app/uwsgi.ini

ENTRYPOINT /docker-entrypoint.sh $0 $@

CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
