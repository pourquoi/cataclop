from django.db import models
from . import managers

class Race(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    start_at = models.DateTimeField()

    num = models.SmallIntegerField()

    num_bis = models.SmallIntegerField(null=True)

    category = models.CharField(max_length=50)
    sub_category = models.CharField(max_length=50)

    condition_age = models.CharField(max_length=50, null=True)
    condition_sex = models.CharField(max_length=50, null=True)

    prize = models.IntegerField()

    distance = models.IntegerField()

    declared_player_count = models.SmallIntegerField()

    session = models.ForeignKey('RaceSession', on_delete=models.CASCADE)

    def __str__(self):
        return '{} R{}C{} {}'.format(self.start_at.strftime('%Y-%m-%d'), self.session.num, self.num, self.start_at.strftime('%H:%M'))


class RaceSession(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    num = models.SmallIntegerField()
    date = models.DateField()

    hippodrome = models.ForeignKey('Hippodrome', on_delete=models.CASCADE)

    objects = managers.RaceSessionQuerySet.as_manager()

    def __str__(self):
        return '{} R{} {}'.format(self.date, self.num, self.hippodrome)


class Player(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    race = models.ForeignKey('Race', on_delete=models.CASCADE)

    age = models.SmallIntegerField()

    is_racing = models.BooleanField(default=True)
    is_first_timer = models.BooleanField(default=False)

    race_count = models.SmallIntegerField()

    victory_count = models.SmallIntegerField()

    placed_count = models.SmallIntegerField() # count of position between 1 and 6th ?
    placed_2_count = models.SmallIntegerField() # count of position 2
    placed_3_count = models.SmallIntegerField() # count of position 3

    earnings = models.IntegerField()
    victory_earnings = models.IntegerField()
    placed_earnings = models.IntegerField()
    year_earnings = models.IntegerField()
    prev_year_earnings = models.IntegerField()

    post_position = models.SmallIntegerField(null=True)

    position = models.SmallIntegerField(null=True)

    handicap_weight = models.SmallIntegerField(null=True)
    handicap_distance = models.IntegerField(null=True)

    time = models.IntegerField(null=True)

    horse = models.ForeignKey('Horse', on_delete=models.CASCADE)
    trainer = models.ForeignKey('Trainer', on_delete=models.CASCADE)
    jockey = models.ForeignKey('Jockey', on_delete=models.CASCADE)
    herder = models.ForeignKey('Herder', on_delete=models.CASCADE, null=True)
    owner = models.ForeignKey('Owner', on_delete=models.CASCADE, null=True)

    def __str__(self):
        return '#{} - {}'.format(self.post_position, self.horse)

class Odds(models.Model):
    imported_at = models.DateTimeField(auto_now=True)
    
    value = models.FloatField()
    evolution = models.FloatField()
    date = models.DateTimeField()

    is_final = models.BooleanField(default=False)
    is_final_ref = models.BooleanField(default=False)

    player = models.ForeignKey('Player', on_delete=models.CASCADE)

    def __str__(self):
        return self.value

    class Meta():
        verbose_name_plural = 'odds'


class Hippodrome(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    code = models.CharField(max_length=10)
    name = models.CharField(max_length=100)
    country = models.CharField(max_length=3)

    def __str__(self):
        return '{} ({})'.format(self.name, self.country)


class Jockey(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Trainer(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Herder(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Owner(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Horse(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100)
    sex = models.CharField(max_length=20)
    breed = models.CharField(max_length=100, null=True)

    father = models.CharField(max_length=100, null=True)
    mother = models.CharField(max_length=100, null=True)
    mother_father = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.name