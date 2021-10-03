import factory
import datetime
from faker import Faker

fake = Faker()


class HorseFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Horse'

    name = factory.Sequence(lambda n: 'horse%s' % n)
    sex = 'MALE'


class OwnerFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Owner'

    name = factory.Sequence(lambda n: 'horse%s' % n)


class HerderFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Herder'

    name = factory.Sequence(lambda n: 'horse%s' % n)


class JockeyFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Jockey'

    name = factory.Sequence(lambda n: 'horse%s' % n)


class TrainerFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Trainer'

    name = factory.Sequence(lambda n: 'horse%s' % n)


class HippodromeFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Hippodrome'

    code = 'CHA'
    name = 'CHANTILLY'
    country = 'FRA'


class RaceSessionFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.RaceSession'

    num = 1
    date = datetime.date.today()
    hippodrome = factory.SubFactory(HippodromeFactory)


class RaceFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Race'

    start_at = datetime.datetime.now()
    num = 1
    session = factory.SubFactory(RaceSessionFactory)

    category = 'PLAT'
    sub_category = 'HANDICAP'

    prize = 1000

    distance = 2500

    declared_player_count = 10


class PlayerFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'core.Player'

    num = 1

    race = factory.SubFactory(RaceFactory)
    horse = factory.SubFactory(HorseFactory)
    trainer = factory.SubFactory(TrainerFactory)
    owner = factory.SubFactory(OwnerFactory)
    jockey = factory.SubFactory(JockeyFactory)
    herder = factory.SubFactory(HerderFactory)

    music = '1p'
    age = 2
    race_count = 1
    victory_count = 1
    placed_count = 1
    placed_2_count = 0
    placed_3_count = 0

    earnings = 100
    victory_earnings = 100
    placed_earnings = 100
    year_earnings = 100
    prev_year_earnings = 100

    post_position = 1
