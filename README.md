## Cataclop

horse race supervised learning and autonomous betting

### EXPLORATION

- [exploration](notebooks/exploration.ipynb)
- [onboarding](notebooks/onboarding.ipynb)

### 1/ INSTALL

```console
cp .env.dist .env
```

```console
docker-compose up -d
```

### 2/ OBTAIN THE JSON RACES FILES

* option 1 : extract an existing archive

```
tar xzvf var/archives/races-2021-01.tgz -C var/scrap
```

* option 2 : scrapping

eg. scrap all January 2021 races
```
docker exec -it cataclop_app python manage.py scrap "2021-01-01" "2021-01-31"
```

### 3/ IMPORT THE JSON FILES
eg. import Janurary 2021 races
```
docker exec -it cataclop_app pipenv run python manage.py parse "2021-01-*"
```

### 4/ TRAIN MODELS
```
docker exec -it cataclop_app python manage.py shell_plus --notebook
```
open the output link (the one starting with http://127.0.0.1:8888/) in your navigator to access the jupyter notebooks

open the *exploration* notebook to get an overview of the data
open the *onboarding* notebook for a quick start on creating models

### REST API

```console
docker exec -it cataclop_app python manage.py createsuperuser
```

The REST API should be running on http://127.0.0.1:8082/api and admin dashboard on http://127.0.0.1:8082/admin

### (advanced) BETTING

```console
docker exec -it cataclop_app python manage.py bet --loop
```

### Tests
```console
docker exec -it cataclop_app python manage.py test --exclude-tag=integration
```
