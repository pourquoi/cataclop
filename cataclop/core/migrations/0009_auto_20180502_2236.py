# Generated by Django 2.0.4 on 2018-05-02 22:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_auto_20180502_2235'),
    ]

    operations = [
        migrations.AlterField(
            model_name='player',
            name='earnings',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='player',
            name='placed_earnings',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='player',
            name='prev_year_earnings',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='player',
            name='victory_earnings',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='player',
            name='year_earnings',
            field=models.IntegerField(),
        ),
    ]
