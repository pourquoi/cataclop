from django.db import models
from . import managers

class Race(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    start_at = models.DateTimeField(db_index=True)

    num = models.SmallIntegerField(db_index=True)

    num_bis = models.SmallIntegerField(null=True)

    category = models.CharField(max_length=50)
    sub_category = models.CharField(max_length=50)

    condition_age = models.CharField(max_length=50, null=True)
    condition_sex = models.CharField(max_length=50, null=True)

    prize = models.IntegerField()

    distance = models.IntegerField()

    declared_player_count = models.SmallIntegerField()

    session = models.ForeignKey('RaceSession', on_delete=models.CASCADE)

    def get_player(self, num: int):
        """
        Get a race player by a his number
        """
        player = None
        try:
            player = next( p for p in self.player_set.all() if p.num == num )
        except StopIteration:
            pass

        return player

    def __str__(self):
        return '{} R{}C{} {}'.format(self.start_at.strftime('%Y-%m-%d'), self.session.num, self.num, self.start_at.strftime('%H:%M'))


class RaceSession(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    num = models.SmallIntegerField(db_index=True)
    date = models.DateField(db_index=True)

    hippodrome = models.ForeignKey('Hippodrome', on_delete=models.CASCADE)

    objects = managers.RaceSessionQuerySet.as_manager()

    def __str__(self):
        return '{} R{} {}'.format(self.date, self.num, self.hippodrome)


class Player(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    race = models.ForeignKey('Race', on_delete=models.CASCADE)

    age = models.SmallIntegerField()

    num = models.SmallIntegerField(db_index=True)

    music = models.CharField(max_length=100)

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

    winner_dividend = models.IntegerField(null=True)
    winner_dividend_offline = models.IntegerField(null=True)
    
    placed_dividend = models.IntegerField(null=True)
    placed_dividend_offline = models.IntegerField(null=True)

    final_odds = models.FloatField(null=True)
    final_odds_offline = models.FloatField(null=True)
    final_odds_unibet = models.FloatField(null=True)

    final_odds_ref = models.FloatField(null=True)
    final_odds_ref_offline = models.FloatField(null=True)
    final_odds_ref_unibet = models.FloatField(null=True)

    horse = models.ForeignKey('Horse', on_delete=models.CASCADE)
    trainer = models.ForeignKey('Trainer', on_delete=models.CASCADE)
    jockey = models.ForeignKey('Jockey', on_delete=models.CASCADE)
    herder = models.ForeignKey('Herder', on_delete=models.CASCADE, null=True)
    owner = models.ForeignKey('Owner', on_delete=models.CASCADE, null=True)

    def __str__(self):
        return '#{} ({}) - {}'.format(self.num, self.final_odds_ref, self.horse)

class Odds(models.Model):
    imported_at = models.DateTimeField(auto_now=True)
    
    value = models.FloatField()
    evolution = models.FloatField()
    date = models.DateTimeField()

    offline = models.BooleanField(default=False)

    is_final = models.BooleanField(default=False)
    is_final_ref = models.BooleanField(default=False)

    player = models.ForeignKey('Player', on_delete=models.CASCADE)

    def __str__(self):
        return "{:f}".format(self.value)

    class Meta():
        verbose_name_plural = 'odds'

class BetResult(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    combo = models.TextField()
    dividend = models.IntegerField()

    type = models.CharField(max_length=100)
    race = models.ForeignKey('Race', on_delete=models.CASCADE)


class Hippodrome(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    code = models.CharField(max_length=10, db_index=True)
    name = models.CharField(max_length=100)
    country = models.CharField(max_length=3, db_index=True)

    def __str__(self):
        return '{} ({})'.format(self.name, self.country)


class Jockey(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return self.name


class Trainer(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return self.name


class Herder(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return self.name



class Owner(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return self.name


class Horse(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)
    sex = models.CharField(max_length=20)
    breed = models.CharField(max_length=100, null=True)

    father = models.CharField(max_length=100, null=True)
    mother = models.CharField(max_length=100, null=True)
    mother_father = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.name