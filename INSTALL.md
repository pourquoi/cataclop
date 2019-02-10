

```console
pipenv install
npm i
```

```console
cp cataclop/settings.dist.py cataclop/settings.py
vim ataclop/settings.py
```

```console
cp scripts/bet_config.json.dist scripts/bet_config.json
vim scripts/bet_config.json
```

```console
vim my.cnf
```
### INSTALL

```conf
[client]
database = DATABASE_NAME
user = USER
password = PASSWORD
default-character-set = utf8
socket = /var/run/mysqld/mysqld.sock
```

```sql
CREATE DATABASE `cataclop-django` /*!40100 DEFAULT CHARACTER SET utf8 */ 
GRANT ALL PRIVILEGES ON `cataclop-django`.* TO 'cataclop'@'localhost';
```

```console
pipenv shell
python manage.py migrate
```

### BETTING

```console
pipenv run python manage.py bet --loop
```

### WORKFLOW

edit a program/model/dataset in cataclop.ml.pipeline

start jupyter
```console
pipenv shell
python manage.py shell_plus --notebook
```

open the train.ipynb notebook
load and train the program
simulate the bets
lock the program (and eventually transfer the model repository a remote machine)


do final edits of the created program and model python files
add the program name to the bet command