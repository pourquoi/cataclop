# Generated by Django 2.0.5 on 2018-05-21 15:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0012_auto_20180519_1641'),
    ]

    operations = [
        migrations.AddField(
            model_name='player',
            name='final_odds_ref',
            field=models.FloatField(null=True),
        ),
    ]
