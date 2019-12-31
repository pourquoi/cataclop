import os
import json
import datetime
import pytz
import glob
from pytz import timezone

from django.db import transaction

from django.core.exceptions import ObjectDoesNotExist

from cataclop.core.models import *

from .scrapper import UnibetScrapper

import logging
logger = logging.getLogger(__name__)

class Parser:

    def __init__(self, root_dir, **kwargs):

        self.parsers = []

        self.parsers.append(PmuParser(root_dir, **kwargs))
        self.parsers.append(UnibetParser(root_dir, **kwargs))

    def parse(self, date=None, **kwargs):

        for parser in self.parsers:
            parser.parse(date=date, **kwargs)


class PmuParser:

    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir

        self.fast = kwargs.get('fast', False)
        self.dry_run = kwargs.get('dry_run', False)

    def parseMissingDividend(self):
        qs = Player.objects.filter(position=1, winner_dividend__isnull=True).prefetch_related('race').values('race__start_at__date').distinct()

        for row in qs:
            self.parse(row['race__start_at__date'].strftime('%Y-%m-%d'))

    def parse(self, date=None, with_offline=True, **kwargs):
        if date is None:
            date = datetime.date.today().isoformat()

        with open(os.path.join(self.root_dir, date, 'programme.json')) as json_data:
            p = json.load(json_data)

        p = p['programme']

        sessions = []

        for rs in p['reunions']:
            try:
                sessions.append(self.importRaceSession(rs, with_offline=with_offline))
            except Exception as err:
                logger.error(err)
                pass

        return sessions


    @transaction.atomic
    def importRaceSession(self, rs, with_offline=False):
        try:
            race_session = RaceSession.objects.get(num=rs['numOfficiel'], date=datetime.date.fromtimestamp(rs['dateReunion']/1000))
        except ObjectDoesNotExist:
            race_session = RaceSession(num=rs['numOfficiel'], date=datetime.date.fromtimestamp(rs['dateReunion']/1000))

        try:
            hippodrome = Hippodrome.objects.get(code=rs['hippodrome']['code'], country=rs['pays']['code'])
        except ObjectDoesNotExist:
            hippodrome = Hippodrome(name=rs['hippodrome']['libelleCourt'], code=rs['hippodrome']['code'], country=rs['pays']['code'])
            if not self.dry_run:
                hippodrome.save()

        race_session.hippodrome = hippodrome

        if not self.dry_run:
            race_session.save()

        for r in rs['courses']:
            race = self.importRace(r, race_session, with_offline=with_offline)
            if race is not None:
                self.importOddsEvolution(r, race_session)

        return race_session


    def importOddsEvolution(self, r, session, offline=False):

        pattern = os.path.join(self.root_dir, session.date.isoformat(), 'R{}C{}-evolution'.format(session.num, r['numOrdre']), 't-minus-[0-9]*')

        odds_files = glob.glob(pattern)

        try:
            race = Race.objects.get(session=session, start_at__date=session.date, num=r['numOrdre'])
        except ObjectDoesNotExist:
            return

        for f in odds_files:
            try:
                with open(f) as json_data:
                    players = json.load(json_data)

                    for p in players['participants']:

                        player = race.get_player(p['numPmu'])

                        if player is None:
                            continue

                        if p.get('dernierRapportDirect'):
                            self.importOdds(p['dernierRapportDirect'], player, is_final=False, offline=False)

                        if p.get('dernierRapportReference'):
                            self.importOdds(p['dernierRapportReference'], player, is_final_ref=False, offline=False)

            except FileNotFoundError:
                pass
                    

    def importRace(self, r, session, with_offline=False):

        try:
            race = Race.objects.get(session=session, start_at__date=session.date, num=r['numOrdre'])
        except ObjectDoesNotExist:
            race = Race(session=session, num=r['numOrdre'])

        race.start_at = datetime.datetime.fromtimestamp(r['heureDepart']/1000)

        race.category = r['discipline'].upper()

        if r.get('categorieParticularite'):
            race.sub_category = r['categorieParticularite'].upper()

        if r['numCourseDedoublee'] != 0:
            race.num_bis = r['numCourseDedoublee']

        race.prize = r['montantPrix']

        if r.get('conditionAge'):
            race.condition_age = r['conditionAge'].upper()

        if r.get('conditionSexe'):
            race.condition_sex = r['conditionSexe'].upper()

        race.declared_player_count = r['nombreDeclaresPartants']

        if r['distanceUnit'].upper() != 'METRE':
            raise ValueError('race {} distance unit not supported: {}'.format(race, r['distanceUnit']))

        race.distance = r['distance']

        try:
            with open(os.path.join(self.root_dir, session.date.isoformat(), 'R{}C{}.json'.format(session.num, race.num))) as json_data:
                players = json.load(json_data)
        except FileNotFoundError:
            return None

        if not self.dry_run:
            race.save()

        try:
            for p in players['participants']:
                player = self.importPlayer(p, race)

                if not player:
                    continue

                player.odds_set.filter(offline=False).delete()

                if p.get('dernierRapportDirect'):
                    self.importOdds(p['dernierRapportDirect'], player, is_final=True, offline=False)

                if p.get('dernierRapportReference'):
                    self.importOdds(p['dernierRapportReference'], player, is_final_ref=True, offline=False)
        except Exception as err:
            logger.error(err)
            race.delete()
            return None

        if with_offline:
            players = None
            try:
                with open(os.path.join(self.root_dir, session.date.isoformat(), 'R{}C{}.offline.json'.format(session.num, race.num))) as json_data:
                    players = json.load(json_data)
            except FileNotFoundError:
                pass

            if players is not None:
                try:
                    for p in players['participants']:

                        player = self.importPlayer(p, race)

                        if not player:
                            continue

                        player.odds_set.filter(offline=True).delete()

                        if p.get('dernierRapportDirect'):
                            self.importOdds(p['dernierRapportDirect'], player, is_final=True, offline=True)

                        if p.get('dernierRapportReference'):
                            self.importOdds(p['dernierRapportReference'], player, is_final_ref=True, offline=True)
                except Exception as err:
                    logger.error(err)
                    pass

        if not self.fast:
            race.betresult_set.all().delete()

            try:
                with open(os.path.join(self.root_dir, session.date.isoformat(), 'R{}C{}-odds.json'.format(session.num, race.num))) as json_data:
                    bet_results = json.load(json_data)
                    for r in bet_results:
                        self.importBetResult(r, race)
            except FileNotFoundError:
                pass

        return race


    def importPlayer(self, p, race):

        try:
            horse = Horse.objects.get(name=p['nom'].upper(), sex=p['sexe'].upper(), breed=p['race'].upper())
        except ObjectDoesNotExist:
            horse = Horse(name=p['nom'].upper(), sex=p['sexe'].upper(), breed=p['race'].upper())

            if p.get('nomPere'):
                horse.father = p['nomPere']
            if p.get('nomMere'):
                horse.mother = p['nomMere']
            if p.get('nomPereMere'):
                horse.mother_father = p['nomPereMere']

            if not self.dry_run:
                horse.save()

        if not p.get('proprietaire'):
            owner = None
        else:
            try:
                owner = Owner.objects.get(name=p['proprietaire'].upper())
            except ObjectDoesNotExist:
                owner = Owner(name=p['proprietaire'].upper())
                if not self.dry_run:
                    owner.save()

        try:
            jockey = Jockey.objects.get(name=p['driver'].upper())
        except ObjectDoesNotExist:
            jockey = Jockey(name=p['driver'].upper())
            if not self.dry_run:
                jockey.save()

        try:
            trainer = Trainer.objects.get(name=p['entraineur'].upper())
        except ObjectDoesNotExist:
            trainer = Trainer(name=p['entraineur'].upper())
            if not self.dry_run:
                trainer.save()

        if not p.get('eleveur'):
            herder = None
        else:
            try:
                herder = Herder.objects.get(name=p['eleveur'].upper())
            except ObjectDoesNotExist:
                herder = Herder(name=p['eleveur'].upper())
                if not self.dry_run:
                    herder.save()

        try:
            player = Player.objects.get(race=race, horse=horse, trainer=trainer, jockey=jockey)
            if self.fast:
                return player
        except ObjectDoesNotExist:
            player = Player(race=race, horse=horse, trainer=trainer, jockey=jockey)

        player.race = race

        player.herder = herder
        player.owner = owner

        #player.final_odds_ref = None
        player.winner_dividend = None
        player.placed_dividend = None

        player.age = p['age']

        player.num = p['numPmu']

        player.post_position = p.get('placeCorde', p['numPmu'])

        player.position = p.get('ordreArrivee')

        player.is_racing = p['statut'].upper() == 'PARTANT'

        player.is_first_timer = p['indicateurInedit']

        player.music = p['musique']

        player.race_count = p['nombreCourses']
        player.victory_count = p['nombreVictoires']
        player.placed_count = p['nombrePlaces']
        player.placed_2_count = p['nombrePlacesSecond']
        player.placed_3_count = p['nombrePlacesTroisieme']

        earnings = p['gainsParticipant']
        player.earnings = earnings.get('gainsCarriere', 0)
        player.victory_earnings = earnings.get('gainsCarriere', 0)
        player.placed_earnings = earnings.get('gainsPlace', 0)
        player.year_earnings = earnings.get('gainsAnneeEnCours', 0)
        player.prev_year_earnings = earnings.get('gainsAnneePrecedente', 0)

        if p.get('handicapValeur'):
            player.handicap_weight = p['handicapValeur']

        if p.get('handicapDistance'):
            player.handicap_distance = p['handicapDistance']

        if p.get('tempsObtenu'):
            player.time = p['tempsObtenu']

        if not self.dry_run:
            player.save()

        return player

    def importOdds(self, o, player, is_final=False, is_final_ref=False, offline=False):
        
        odds = Odds(value=o['rapport'], is_final=is_final, is_final_ref=is_final_ref)
        odds.evolution = o.get('nombreIndicateurTendance', 0)
        odds.date = datetime.datetime.fromtimestamp(o['dateRapport']/1000)
        odds.player = player
        odds.offline = offline

        odds.save()

        if is_final_ref:
            if offline:
                player.final_odds_ref_offline = odds.value
            else:
                player.final_odds_ref = odds.value
            player.save()
        elif is_final:
            if offline:
                player.final_odds_offline = odds.value
            else:
                player.final_odds = odds.value
            player.save()

        return odds

    def importBetResult(self, r, race):

        for rr in r['rapports']:
            if r['typePari'] not in ['SIMPLE_GAGNANT', 'SIMPLE_PLACE', 'E_SIMPLE_PLACE', 'E_SIMPLE_GAGNANT']:
                continue

            result = BetResult(type=r['typePari'], combo=json.dumps(rr['combinaison']), dividend=rr['dividendePourUnEuro'])
            result.race = race

            result.save()

            for n in rr['combinaison']:
                p = race.get_player(n)

                if p is None:
                    continue

                if r['typePari'] == 'E_SIMPLE_GAGNANT':
                    p.winner_dividend = result.dividend
                    p.save()
                elif r['typePari'] == 'E_SIMPLE_PLACE':
                    p.placed_dividend = result.dividend
                    p.save()


class UnibetParser:

    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir

        self.fast = kwargs.get('fast', False)
        self.dry_run = kwargs.get('dry_run', False)

    def parse(self, date=None, **kwargs):
        if date is None:
            date = datetime.date.today().isoformat()
        
        path = os.path.join(self.root_dir, date, 'programme.{}.json'.format(UnibetScrapper.name))

        if not os.path.isfile(path):
            return

        with open(path) as json_data:
            sessions = json.load(json_data)

        for rs in sessions:

            race_session = None

            try:
                race_session = RaceSession.objects.get(num=rs['rank'], date=datetime.date.fromtimestamp(rs['date']/1000))
            except ObjectDoesNotExist:
                continue
            except Exception as err:
                logger.error(err)
                continue

            for r in rs['races']:

                race = None

                try:
                    race = Race.objects.get(session=race_session, start_at__date=race_session.date, num=r['rank'])
                except ObjectDoesNotExist:
                    continue

                race_name = 'R{}C{}'.format(rs['rank'], r['rank'])

                race_file = os.path.join(self.root_dir, race_session.date.isoformat(), race_name + '.' + UnibetScrapper.name + '.json')

                if not os.path.isfile(race_file):
                    continue

                with open(race_file) as json_data:
                    r_details = json.load(json_data)

                for n, odds in r_details['details']['probables']['5'].items():

                    runner = None
                    for p in r_details['runners']:
                        if p['rank'] == int(n):
                            runner = p
                            break

                    player = race.get_player( int(n) )

                    if player is not None:
                        player.final_odds_ref_unibet = odds
                        player.save()

                for n, odds in r_details['details']['probables']['6'].items():

                    runner = None
                    for p in r_details['runners']:
                        if p['rank'] == int(n):
                            runner = p
                            break

                    player = race.get_player( int(n) )

                    if player is not None:
                        player.final_odds_unibet = odds
                        player.save()

