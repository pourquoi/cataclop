#!/bin/sh
cd /home/cataclop/cataclop
python manage.py scrap
python manage.py parse --predict
python manage.py scrap tomorrow tomorrow
python manage.py parse tomorrow --predict