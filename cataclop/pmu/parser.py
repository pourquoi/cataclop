import os
import json
import datetime
import pytz
from pytz import timezone

from django.db import transaction

from django.core.exceptions import ObjectDoesNotExist

from cataclop.core.models import *

class Parser:

    dry_run = False

    def __init__(self, root_dir):
        self.root_dir = root_dir


    def parse(self, date):
        with open(os.path.join(self.root_dir, date, 'programme.json')) as json_data:
            p = json.load(json_data)

        p = p['programme']

        sessions = []

        for rs in p['reunions']:
            sessions.append(self.importRaceSession(rs))

        return sessions


    @transaction.atomic
    def importRaceSession(self, rs):
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
            self.importRace(r, race_session)

        return race_session


    def importRace(self, r, session):

        print('R{}C{}'.format(session.num, r['numOrdre']))
        
        try:
            race = Race.objects.get(session=session, start_at__date=session.date, num=r['numOrdre'])
        except ObjectDoesNotExist:
            race = Race(session=session, num=r['numOrdre'])

        race.start_at = datetime.datetime.fromtimestamp(r['heureDepart']/1000)

        race.category = r['discipline'].upper()
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

        for p in players['participants']:
            player = self.importPlayer(p, race)

            if not player:
                continue

            player.odds_set.all().delete()

            if p.get('dernierRapportDirect'):
                self.importOdds(p['dernierRapportDirect'], player, is_final=True)

            if p.get('dernierRapportReference'):
                self.importOdds(p['dernierRapportReference'], player, is_final_ref=True)

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
            player = Player.objects.get(horse=horse, trainer=trainer, jockey=jockey)
        except ObjectDoesNotExist:
            player = Player(horse=horse, trainer=trainer, jockey=jockey)

        if not p.get('placeCorde'):
            if player.pk:
                player.delete()
            return None

        player.race = race

        player.herder = herder
        player.owner = owner

        player.final_odds_ref = None
        player.winner_dividend = None
        player.placed_dividend = None

        player.age = p['age']

        player.num = p['numPmu']

        player.post_position = p['placeCorde']

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

    def importOdds(self, o, player, is_final=False, is_final_ref=False):
        
        odds = Odds(value=o['rapport'], is_final=is_final, is_final_ref=is_final_ref)
        odds.evolution = o['nombreIndicateurTendance']
        odds.date = datetime.datetime.fromtimestamp(o['dateRapport']/1000)
        odds.player = player

        odds.save()

        if is_final_ref:
            player.final_odds_ref = odds.value
            player.save()
        elif is_final:
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







