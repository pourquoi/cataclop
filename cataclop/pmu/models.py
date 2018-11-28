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

    def __str__(self):
        return '{}'.format(self.url)