# Generated by Django 2.0.4 on 2018-04-27 10:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_racesession_hippodrome'),
    ]

    operations = [
        migrations.AddField(
            model_name='racesession',
            name='num',
            field=models.SmallIntegerField(default=None),
            preserve_default=False,
        ),
    ]
