#!/bin/sh
cd /home/cataclop/cataclop
pipenv run python manage.py scrap
pipenv run python manage.py parse --predict
pipenv run python manage.py scrap tomorrow tomorrow
pipenv run python manage.py parse tomorrow --predict