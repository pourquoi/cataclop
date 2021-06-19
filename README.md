## Cataclop

horse race supervised learning and autonomous betting

### EXPLORATION

[notebook](notebooks/exploration.ipynb)

### 1/ INSTALL

```console
cp cataclop/settings.dist.py cataclop/settings.py
cp .env.dist .env
```

```console
docker-compose up -d
```

```console
docker exec -it cataclop_app pipenv run python manage.py migrate
```

### 2/ IMPORT DATA

#### get the daily races JSON files
option 1 : extract an existing archive

```
tar xzvf var/races-2021-01.tgz -C var/scrap
```

option 2 : scrapping

eg. scrap all January 2021 races
```
docker exec -it cataclop_app pipenv run python manage.py scrap "2021-01-01" "2021-01-31"
```

#### import the JSON files
eg. import Janurary 2021 races
```
docker exec -it cataclop_app pipenv run python manage.py parse "2021-01-*"
```

### 3/ TRAIN MODELS
```
docker exec -it cataclop_app pipenv run python manage.py shell_plus --notebook
```
click on the output link to access the jupyter notebooks

open the *exploration* notebook to get an overview of the data
open the *onboarding* notebook for a quick start on creating models

### (optionnal) RUN REST API
```
docker exec -it cataclop_app pipenv run python manage.py runserver_plus 0.0.0.0:8082
```

### (advanced) BETTING

```console
cp scripts/bet_config.json.dist scripts/bet_config.json
vim scripts/bet_config.json
```

```console
pipenv run python manage.py bet --loop
```
