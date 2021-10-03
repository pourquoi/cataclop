from cataclop.core.models import *
from django.db.models import Avg, Case, Count, Sum
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def compute_players_stats():
    players = Player.objects.all()
    for p in players:
        compute_player_stats(p)


def compute_player_stats(p, force=False):
    if p.trainer_winning_rate is not None and not force:
        return

    same_trainer_players = Player.objects.filter(trainer=p.trainer, imported_at__lt=p.imported_at).exclude(
        horse=p.horse)
    stats = same_trainer_players.aggregate(winner_dividend_sum=Sum('winner_dividend'), c=Count('id'),
                                           wins=Count('winner_dividend'))

    if stats['c'] == 0:
        return

    if stats['wins'] is None:
        stats['wins'] = 0
    if stats['winner_dividend_sum'] is None:
        stats['winner_dividend_sum'] = 0
    p.trainer_winning_rate = stats['wins'] / stats['c']
    p.trainer_avg_winning_dividend = (stats['winner_dividend_sum'] / 100. - stats['c']) / stats['c']

    history = list(Player.objects.filter(horse=p.horse, imported_at__lt=p.imported_at).order_by('-imported_at')[0:5])
    if len(history):
        p.hist_1_days = (p.race.start_at - history[0].race.start_at).days
    if len(history) > 1:
        p.hist_2_days = (p.race.start_at - history[1].race.start_at).days
    if len(history) > 2:
        p.hist_3_days = (p.race.start_at - history[2].race.start_at).days

    if len(history):
        p.jockey_change = p.jockey != history[0].jockey

    p.save()


# https://trueskill.org/
def compute_races_trueskill(force=False, **kwargs):
    from cataclop.core.paginator import CursorPaginator
    print(kwargs)
    page_size = 100
    if len(kwargs):
        qs = Race.objects.filter(**kwargs)
    else:
        qs = Race.objects.all()

    paginator = CursorPaginator(qs, ordering=('id',))
    after = None

    while True:
        page = paginator.page(after=after, first=page_size)
        if page:
            for race in page.items:
                print(race.start_at)
                compute_race_trueskill(race, force=force)
        else:
            return

        if not page.has_next:
            break

        after = paginator.cursor(instance=page[-1])


def _get_related_races(race, races, depth=0, max_depth=2):
    for p in race.player_set.all():
        history = reversed(
            list(Player.objects.filter(horse=p.horse, imported_at__lt=p.imported_at).order_by('-imported_at')[0:10]))
        for h in history:
            races.append(h.race)
            if depth < max_depth:
                _get_related_races(h.race, races, depth + 1)


def get_related_races(race):
    races = []
    _get_related_races(race, races, 0, 0)
    return races


def compute_race_trueskill(race, force=False):
    from trueskill import Rating, quality, rate

    ratings = {}
    races_seen = []

    for p in race.player_set.all():

        if p.trueskill_mu is not None and not force:
            return

        if p.horse.id not in ratings:
            ratings[p.horse.id] = Rating()

    races = get_related_races(race)

    for r in races:
        races_seen.append(r.id)
        teams = []
        ranks = []
        players = list(r.player_set.all())
        for p in players:
            if p.horse.id not in ratings:
                ratings[p.horse.id] = Rating()
            teams.append((ratings[p.horse.id],))
            rank = p.position if p.position and (p.position < 10 and p.position > 0) else 10
            ranks.append(rank)

        if len(teams) < 2:
            continue

        res = rate(teams, ranks)
        for z in zip(players, res):
            ratings[z[0].horse.id] = z[1][0]

    for p in race.player_set.all():

        if p.horse.id not in ratings:
            continue

        p.trueskill_mu = ratings[p.horse.id].mu
        p.trueskill_sigma = ratings[p.horse.id].sigma
        p.save()



def deduplicate_scrap_data():
    races = Race.objects.all().values('num', 'session__num', 'start_at__date', 'session__hippodrome_id').annotate(
        cnt=Count('num')).filter(cnt__gte=2)

    logger.debug('found {} races to deduplicate'.format(len(races)))

    for race in races:
        t = list(Race.objects.filter(num=race['num'], session__num=race['session__num'],
                                     start_at__date=race['start_at__date'],
                                     session__hippodrome_id=race['session__hippodrome_id']))
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
