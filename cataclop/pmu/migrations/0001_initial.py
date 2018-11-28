# Generated by Django 2.1.3 on 2018-11-24 22:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('core', '0016_player_music'),
    ]

    operations = [
        migrations.CreateModel(
            name='Bet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now=True)),
                ('url', models.TextField(null=True)),
                ('simulation', models.BooleanField()),
                ('returncode', models.IntegerField(null=True)),
                ('stderr', models.TextField(null=True)),
                ('stdout', models.TextField(null=True)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=10, null=True)),
                ('returns', models.DecimalField(decimal_places=2, max_digits=10, null=True)),
                ('player', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Player')),
            ],
        ),
    ]
