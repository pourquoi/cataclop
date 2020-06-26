from rest_framework import serializers

from cataclop.users.models import User
from cataclop.core.models import RaceSession, Race, Player, Hippodrome, Horse, Trainer, Herder, Owner, Jockey
from cataclop.pmu.models import Bet


class FullUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups', 'is_staff', 'is_superuser', 'date_joined', 'last_login')

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'date_joined')

class HippodromeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hippodrome
        fields = '__all__'

class HorseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Horse
        fields = '__all__'

class TrainerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trainer
        fields = '__all__'

class JockeySerializer(serializers.ModelSerializer):
    class Meta:
        model = Jockey
        fields = '__all__'

class OwnerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Owner
        fields = '__all__'

class HerderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Herder
        fields = '__all__'

class PlayerSerializer(serializers.ModelSerializer):
    owner = OwnerSerializer(read_only=True)
    herder = HerderSerializer(read_only=True)
    jockey = JockeySerializer(read_only=True)
    horse = HorseSerializer(read_only=True)
    
    class Meta:
        model = Player
        fields = '__all__'

class SimpleRaceSessionSerializer(serializers.ModelSerializer):
    hippodrome = HippodromeSerializer(read_only=True)

    class Meta:
        model = RaceSession
        fields = ('id', 'num', 'hippodrome')

class SimpleRaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Race
        fields = ('id', 'num', 'start_at', 'declared_player_count', 'sub_category', 'category', 'prize')
    
    category = serializers.CharField(source='get_category_label')
    sub_category = serializers.CharField(source='get_sub_category_label')


class RaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Race
        fields = '__all__'

    player_set = PlayerSerializer(many=True, read_only=True)

    session = SimpleRaceSessionSerializer(read_only=True)

    category = serializers.CharField(source='get_category_label')
    sub_category = serializers.CharField(source='get_sub_category_label')
    condition_sex = serializers.CharField(source='get_condition_sex_label')
    condition_age = serializers.CharField(source='get_condition_age_label')


class ListRaceSerializer(serializers.ModelSerializer):
    session = SimpleRaceSessionSerializer(read_only=True)

    category = serializers.CharField(source='get_category_label')
    sub_category = serializers.CharField(source='get_sub_category_label')

    class Meta:
        model = Race
        fields = ('id', 'num', 'start_at', 'declared_player_count', 'sub_category', 'category', 'prize', 'session')

class RaceSessionSerializer(serializers.ModelSerializer):
    race_set = SimpleRaceSerializer(many=True)

    hippodrome = HippodromeSerializer(read_only=True)

    class Meta:
        model = RaceSession
        fields = '__all__'

class BetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bet
        fields = '__all__'