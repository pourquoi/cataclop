# Generated by Django 2.0.4 on 2018-05-02 22:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0009_auto_20180502_2236'),
    ]

    operations = [
        migrations.AlterField(
            model_name='race',
            name='condition_age',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='race',
            name='condition_sex',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
