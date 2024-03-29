# Generated by Django 3.1.5 on 2021-01-29 00:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0024_odds_whale'),
    ]

    operations = [
        migrations.AlterField(
            model_name='odds',
            name='date',
            field=models.DateTimeField(db_index=True),
        ),
        migrations.AlterField(
            model_name='odds',
            name='is_final',
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AlterField(
            model_name='odds',
            name='is_final_ref',
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AlterField(
            model_name='odds',
            name='offline',
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AlterField(
            model_name='odds',
            name='whale',
            field=models.BooleanField(db_index=True, default=False),
        ),
    ]
