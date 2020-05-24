#!/bin/sh

while [ 1 ]
do
  pipenv run python manage.py bet
  sleep 20
done
