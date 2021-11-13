import string
from django.db import models
from . import managers


CONDITIONS_AGE = (
    { "id": "DEUX_ANS", "label": "2 ans" },
    { "id": "TROIS_ANS", "label": "3 ans" },
    { "id": "QUATRE_ANS", "label": "4 ans" },
    { "id": "CINQ_ANS", "label": "5 ans" },
    { "id": "SIX_ANS", "label": "6 ans" },
    { "id": "DEUX_ANS_ET_PLUS", "label": "2+ ans" },
    { "id": "TROIS_ANS_ET_PLUS", "label": "3+ ans" },
    { "id": "QUATRE_ANS_ET_PLUS", "label": "4+ ans" },
    { "id": "CINQ_ANS_ET_PLUS", "label": "5+ ans" },
    { "id": "CINQ_SIX_ANS", "label": "5, 6 ans" },
    { "id": "QUATE_CINQ_SIX_ANS", "label": "4, 5, 6 ans" },
    { "id": "DEUX_ET_TROIS_ANS", "label": "2, 3 ans" },
    { "id": "TROIS_QUATRE_CINQ_ANS", "label": "3, 4, 5 ans" },
    { "id": "DEUX_TROIS_QUATRE_ANS", "label": "2, 3, 4 ans" },
    { "id": "INCONNU", "label": ""}
)

CONDITIONS_SEX = (
    { "id": "FEMELLES", "label": "Femelles" },
    { "id": "FEMELLES_ET_MALES", "label": "Femelles et mâles" },
    { "id": "TOUS_CHEVAUX", "label": "Tous" },
    { "id": "MALES_ET_HONGRES", "label": "Mâles et hongres" },
    { "id": "MALES", "label": "Mâles" },
    { "id": "FEMELLES_ET_HONGRES", "label": "Femelles et hongres" },
    { "id": "HONGRES", "label": "Hongres" }
)

CATEGORIES = (
    { "id": "PLAT", "label": "Plat" },
    { "id": "STEEPLECHASE", "label": "Steeple-chase" },
    { "id": "HAIE", "label": "Saut de Haies" },
    { "id": "ATTELE", "label": "Trot Attelé" },
    { "id": "MONTE", "label": "Trot Monté" },
    { "id": "CROSS", "label": "Cross-country" }
)

SUB_CATEGORIES = (
    { "id": "GROUPE_I", "label": "Groupe I" },
    { "id": "GROUPE_II", "label": "Groupe II" },
    { "id": "GROUPE_III", "label": "Groupe III" },
    { "id": "COURSE_A_CONDITIONS", "label": "Course à conditions" },
    { "id": "HANDICAP_DE_CATEGORIE", "label": "Handicap de catégorie" },
    { "id": "HANDICAP_CATEGORIE_DIVISE", "label": "Handicap de catégorie divisé" },
    { "id": "NATIONALE", "label": "Nationale" },
    { "id": "APPRENTIS_LADS_JOCKEYS", "label": "Apprentis" },
    { "id": "HANDICAP", "label": "Handicap" },
    { "id": "AUTOSTART", "label": "Autostart" },
    { "id": "EUROPEENNE_AUTOSTART", "label": "Européenne - autostart" },
    { "id": "EUROPEENNE", "label": "Européenne" },
    { "id": "HANDICAP_DIVISE", "label": "Handicap divisé" },
    { "id": "NATIONALE_AUTOSTART", "label": "Nationale - autostart" },
    { "id": "AMATEURS", "label": "Amateurs" },
    { "id": "A_RECLAMER_AUTOSTART", "label": "A réclamer - autostart" },
    { "id": "APPRENTIS_LADS_JOCKEYS_EUROPEENNE", "label": "Apprentis - européenne" },
    { "id": "INTERNATIONALE", "label": "Internationale" },
    { "id": "INTERNATIONALE_AUTOSTART", "label": "Internationale - autostart" },
    { "id": "AMATEURS_AUTOSTART", "label": "Amateurs - autostart" },
    { "id": "COURSE_A_CONDITION_QUALIF_HP", "label": "Course à conditions" },
    { "id": "APPRENTIS_LADS_JOCKEYS_AUTOSTART", "label": "Apprentis - autostart" },
    { "id": "AMATEURS_NATIONALE", "label": "Amateurs - nationale" },
    { "id": "APPRENTIS_LADS_JOCKEYS_A_RECLAMER_AUTOSTART", "label": "Apprentis à réclamer - autostart" },
    { "id": "QUALIFICATION_ACCAF", "label": "Qualif. ACCAF" },
    { "id": "AMATEURS_INTERNATIONALE_AUTOSTART", "label": "Amateurs - internationale - autostart" },
    { "id": "A_RECLAMER_APPRENTIS_LADS_JOCKEYS", "label": "A réclamer - apprentis" },
    { "id": "FINALE_REGIONALE_ACCAF", "label": "Finale ACCAF" },
    { "id": "COURSE_INTERNATIONALE", "label": "Internationale" },
    { "id": "AMATEURS_EUROPEENNE_AUTOSTART", "label": "Amateurs - européenne - autostart" },
    { "id": "AMATEURS_EUROPEENNE", "label": "Amateurs - européenne" },
    { "id": "AMATEURS_INTERNATIONALE", "label": "Amateurs - internationale" },
    { "id": "A_RECLAMER_AMATEURS", "label": "A réclamer" },
    { "id": "A_RECLAMER_AMATEURS_AUTOSTART", "label": "A réclamer" },
    { "id": "AMATEURS_PRIORITE_AUX_PROPRIETAIRES", "label": "Amateurs - propriétaires" },
    { "id": "COURSE_AP_EUROPEENNE", "label": "" },
    { "id": "A_RECLAMER_EUROPEENNE", "label": "A réclamer - européenne" },
    { "id": "AMATEURS_DAMES_AUTOSTART", "label": "Amateurs - dames" },
    { "id": "INCONNU", "label": "" }
)

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

    def get_category_label(self):
        label = [c["label"] for c in CATEGORIES if c["id"] == self.category]
        return label[0] if len(label) else string.capwords(self.category.lower().replace('_', ' '))

    def get_sub_category_label(self):
        label = [c["label"] for c in SUB_CATEGORIES if c["id"] == self.sub_category]
        return label[0] if len(label) else string.capwords(self.sub_category.lower().replace('_', ' '))

    def get_condition_sex_label(self):
        if self.condition_sex is None:
            return None
        label = [c["label"] for c in CONDITIONS_SEX if c["id"] == self.condition_sex]
        return label[0] if len(label) else string.capwords(self.category.lower().replace('_', ' '))

    def get_condition_age_label(self):
        if self.condition_age is None:
            return None
        label = [c["label"] for c in CONDITIONS_AGE if c["id"] == self.condition_age]
        return label[0] if len(label) else string.capwords(self.category.lower().replace('_', ' '))

    def get_player(self, num: int):
        """
        Get a race player by his number
        """
        player = None
        try:
            player = next( p for p in self.player_set.all() if p.num == num )
        except StopIteration:
            pass

        return player

    def __str__(self):
        return u'{} R{}C{} {}'.format(self.start_at.strftime('%Y-%m-%d'), self.session.num, self.num, self.start_at.strftime('%H:%M'))


class RaceSession(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    num = models.SmallIntegerField(db_index=True)
    date = models.DateField(db_index=True)

    hippodrome = models.ForeignKey('Hippodrome', on_delete=models.CASCADE)

    def __str__(self):
        return u'{} R{} {}'.format(self.date, self.num, self.hippodrome)


class Player(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    race = models.ForeignKey('Race', on_delete=models.CASCADE)

    age = models.SmallIntegerField()

    num = models.SmallIntegerField(db_index=True)

    music = models.CharField(max_length=100)

    is_racing = models.BooleanField(default=True)
    is_first_timer = models.BooleanField(default=False)

    jockey_change = models.BooleanField(default=False)

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

    trueskill_mu = models.FloatField(null=True)
    trueskill_sigma = models.FloatField(null=True)

    time = models.IntegerField(null=True)

    hist_1_days = models.IntegerField(null=True)
    hist_2_days = models.IntegerField(null=True)
    hist_3_days = models.IntegerField(null=True)

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

    trainer_winning_rate = models.FloatField(null=True)
    trainer_avg_winning_dividend = models.FloatField(null=True)

    jockey_winning_rate = models.FloatField(null=True)
    jockey_avg_winning_dividend = models.FloatField(null=True)
    
    herder_winning_rate = models.FloatField(null=True)
    herder_avg_winning_dividend = models.FloatField(null=True)

    position_prediction = models.FloatField(null=True)

    horse = models.ForeignKey('Horse', on_delete=models.CASCADE)
    trainer = models.ForeignKey('Trainer', on_delete=models.CASCADE)
    jockey = models.ForeignKey('Jockey', on_delete=models.CASCADE)
    herder = models.ForeignKey('Herder', on_delete=models.CASCADE, null=True)
    owner = models.ForeignKey('Owner', on_delete=models.CASCADE, null=True)

    def __str__(self):
        return u'#{} ({}) - {}'.format(self.num, self.final_odds_ref, self.horse)


class Odds(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    value = models.FloatField()
    evolution = models.FloatField()
    date = models.DateTimeField(db_index=True,)

    whale = models.BooleanField(default=False, db_index=True,)

    offline = models.BooleanField(default=False, db_index=True,)

    is_final = models.BooleanField(default=False, db_index=True,)
    is_final_ref = models.BooleanField(default=False, db_index=True,)

    player = models.ForeignKey('Player', on_delete=models.CASCADE)

    def __str__(self):
        return u"{:f}".format(self.value)

    class Meta():
        verbose_name_plural = 'odds'


class BetResult(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    combo = models.JSONField()
    dividend = models.IntegerField()

    type = models.CharField(max_length=100)
    race = models.ForeignKey('Race', on_delete=models.CASCADE)


class Hippodrome(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    code = models.CharField(max_length=10, db_index=True)
    name = models.CharField(max_length=100)
    country = models.CharField(max_length=3, db_index=True)

    def __str__(self):
        return u'{} ({})'.format(self.name, self.country)


class Jockey(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return u"{}".format(self.name)


class Trainer(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return u"{}".format(self.name)


class Herder(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return u"{}".format(self.name)


class Owner(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return u"{}".format(self.name)


class Horse(models.Model):
    imported_at = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=100, db_index=True)
    sex = models.CharField(max_length=20)
    breed = models.CharField(max_length=100, null=True)

    father = models.CharField(max_length=100, null=True)
    mother = models.CharField(max_length=100, null=True)
    mother_father = models.CharField(max_length=100, null=True)

    def __str__(self):
        return u"{}".format(self.name)