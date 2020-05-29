from cataclop.core.models import BetResult, Herder, Hippodrome, Horse, Jockey, Odds, Owner, Player, Race, RaceSession, Trainer
from django.db.models import Avg, Case, Count

def deduplicate():

    horses = Horse.objects.all().values('name', 'sex').annotate(cnt=Count('name')).filter(cnt__gte=2)

    for horse in horses: 
        t = list(Horse.objects.filter(name=horse['name'], sex=horse['sex']))
        players = Player.objects.filter(horse=t[0]) 

        i = 1
        while i < len(t):
            players2 = Player.objects.filter(horse=t[i])
 
            for p in players2: 
                p.horse = t[0] 
                p.save()

            t[i].delete()
            i = i + 1

    herders = Herder.objects.all().values('name').annotate(cnt=Count('name')).filter(cnt__gte=2)

    for herder in herders: 
        t = list(Herder.objects.filter(name=herder['name']))
        players = Player.objects.filter(herder=t[0])

        i = 1
        while i < len(t):
            players2 = Player.objects.filter(herder=t[i])
 
            for p in players2: 
                p.herder = t[0] 
                p.save()

            t[i].delete()
            i = i + 1

    owners = Owner.objects.all().values('name').annotate(cnt=Count('name')).filter(cnt__gte=2)

    for owner in owners: 
        t = list(Owner.objects.filter(name=owner['name']))
        players = Player.objects.filter(owner=t[0])

        i = 1
        while i < len(t):
            players2 = Player.objects.filter(owner=t[i])
 
            for p in players2: 
                p.owner = t[0] 
                p.save()

            t[i].delete()
            i = i + 1

    jockeys = Jockey.objects.all().values('name').annotate(cnt=Count('name')).filter(cnt__gte=2)

    for jockey in jockeys: 
        t = list(Jockey.objects.filter(name=jockey['name']))
        players = Player.objects.filter(jockey=t[0])

        i = 1
        while i < len(t):
            players2 = Player.objects.filter(jockey=t[i])
 
            for p in players2: 
                p.jockey = t[0] 
                p.save()

            t[i].delete()
            i = i + 1

    trainers = Trainer.objects.all().values('name').annotate(cnt=Count('name')).filter(cnt__gte=2)

    for trainer in trainers: 
        t = list(Trainer.objects.filter(name=trainer['name']))
        players = Player.objects.filter(trainer=t[0])

        i = 1
        while i < len(t):
            players2 = Player.objects.filter(trainer=t[i])
 
            for p in players2: 
                p.trainer = t[0] 
                p.save()

            t[i].delete()
            i = i + 1