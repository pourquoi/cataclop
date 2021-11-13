import json
import hashlib

from django.db import models


class Model(models.Model):
    created_at = models.DateTimeField(auto_now=True)

    name = models.CharField()


class Train(models.Model):
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField()

    model = models.ForeignKey('Model', on_delete=models.CASCADE)

    dataset_params = models.JSONField(nullable=True)
    model_params = models.JSONField(nullable=True)
    program_params = models.JSONField(nullable=True)