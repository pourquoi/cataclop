from cataclop.core.models import BetResult, Herder, Hippodrome, Horse, Jockey, Odds, Owner, Player, Race, RaceSession, Trainer
from django.db.models import Avg, Case, Count, Sum
from datetime import timedelta

import logging
logger = logging.getLogger(__name__)

def deduplicate():

    races = Race.objects.all().values('num', 'session__num', 'start_at__date', 'session__hippodrome_id').annotate(cnt=Count('num')).filter(cnt__gte=2) 

    logger.debug('found {} races to deduplicate'.format(len(races)))

    for race in races: 
         t = list(Race.objects.filter(num=race['num'], session__num=race['session__num'], start_at__date=race['start_at__date'], session__hippodrome_id=race['session__hippodrome_id'])) 
         i = 1 
         while i < len(t): 
             t[i].delete() 
             i = i + 1 

    horses = Horse.objects.all().values('name', 'sex').annotate(cnt=Count('name')).filter(cnt__gte=2)

    logger.debug('found {} horses to deduplicate'.format(len(horses)))

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

    logger.debug('found {} herders to deduplicate'.format(len(herders)))

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

    logger.debug('found {} owners to deduplicate'.format(len(owners)))

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

    logger.debug('found {} jockeys to deduplicate'.format(len(jockeys)))

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

    logger.debug('found {} trainers to deduplicate'.format(len(trainers)))

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
    
    hippodromes = Hippodrome.objects.all().values('code', 'country').annotate(cnt=Count('code')).filter(cnt__gte=2)

    logger.debug('found {} hippodromes to deduplicate'.format(len(hippodromes)))

    for hippodrome in hippodromes:
        t = list(Hippodrome.objects.filter(code=hippodrome['code'], country=hippodrome['country']))
        sessions = RaceSession.objects.filter(hippodrome=t[0])

        i = 1
        while i < len(t):
            sessions2 = RaceSession.objects.filter(hippodrome=t[i])
 
            for sess in sessions2: 
                sess.hippodrome = t[0] 
                sess.save()

            t[i].delete()
            i = i + 1