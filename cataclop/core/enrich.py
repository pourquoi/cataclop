from cataclop.core.models import BetResult, Herder, Hippodrome, Horse, Jockey, Odds, Owner, Player, Race, RaceSession, Trainer
from django.db.models import Avg, Case, Count, Sum
from datetime import timedelta


def compute_players_stats():
    players = Player.objects.all()
    for p in players:
        compute_player_stats(p)

def compute_player_stats(p, force=False):
    if p.trainer_winning_rate is not None and not force:
        return

    same_trainer_players = Player.objects.filter(trainer=p.trainer, imported_at__lt=p.imported_at).exclude(horse=p.horse) 
    stats = same_trainer_players.aggregate(winner_dividend=Sum('winner_dividend'), c=Count('id'), wins=Count('winner_dividend'))

    if stats['c'] == 0:
        return

    if stats['wins'] is None:
        stats['wins'] = 0
    if stats['winner_dividend'] is None:
        stats['winner_dividend'] = 0
    p.trainer_winning_rate = stats['wins'] / stats['c']
    p.trainer_avg_winning_dividend = (stats['winner_dividend']/100. - stats['c']) / stats['c']

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

    paginator = CursorPaginator(qs, ordering=('id', ))
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

def compute_race_trueskill(race, force=False):
    from trueskill import Rating, quality, rate

    ratings = {}

    races_seen = []

    for p in race.player_set.all():

        if p.trueskill_mu is not None and not force:
            return
        
        if p.horse.id not in ratings:
            ratings[p.horse.id] = Rating()

        history = reversed(list(Player.objects.filter(horse=p.horse, imported_at__lt=p.imported_at).order_by('-imported_at')[0:10]))

        for h in history:
            if h.race.id in races_seen:
                continue
            races_seen.append(h.race.id)
            hplayers = list(h.race.player_set.all())
            teams = []
            ranks = []
            for hp in hplayers:
                if hp.horse.id not in ratings:
                    ratings[hp.horse.id] = Rating()

                teams.append((ratings[hp.horse.id], ))
                rank = hp.position if hp.position and (hp.position < 10 and hp.position > 0) else 10
                ranks.append(rank)

            if len(teams) < 2:
                continue

            res = rate(teams, ranks)

            for r in zip(hplayers, res):
                ratings[r[0].horse.id] = r[1][0]

    for p in race.player_set.all():

        if p.horse.id not in ratings:
            continue

        p.trueskill_mu = ratings[p.horse.id].mu
        p.trueskill_sigma = ratings[p.horse.id].sigma
        p.save()

                

                
