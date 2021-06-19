FROM python:3.7
ENV PYTHONUNBUFFERED=1
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
RUN pip install pipenv
WORKDIR /app
COPY . /app
ENTRYPOINT ["tail", "-f", "/dev/null"]