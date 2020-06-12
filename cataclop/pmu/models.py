from django.db import models

class Bet(models.Model):
    created_at = models.DateTimeField(auto_now=True)

    url = models.TextField(null=True)

    simulation = models.BooleanField()

    returncode = models.IntegerField(null=True)

    stderr = models.TextField(null=True)

    stdout = models.TextField(null=True)

    player = models.ForeignKey('core.Player', null=True, on_delete=models.SET_NULL)

    amount = models.DecimalField(null=True, max_digits=10, decimal_places=2)

    returns = models.DecimalField(null=True, max_digits=10, decimal_places=2)

    program = models.TextField(null=True)

    provider = models.TextField(null=True)

    stats_scrap_time = models.IntegerField(null=True)

    stats_prediction_time = models.IntegerField(null=True)

    stats_bet_time = models.IntegerField(null=True)

    def __str__(self):
        return '{}'.format(self.url)