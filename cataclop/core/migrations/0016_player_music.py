# Generated by Django 2.0.5 on 2018-05-27 20:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0015_player_final_odds'),
    ]

    operations = [
        migrations.AddField(
            model_name='player',
            name='music',
            field=models.CharField(default=None, max_length=100),
            preserve_default=False,
        ),
    ]
