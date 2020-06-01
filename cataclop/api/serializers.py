from rest_framework import serializers

from cataclop.users.models import User
from cataclop.core.models import RaceSession, Race, Player, Hippodrome, Horse, Trainer, Herder, Owner, Jockey
from cataclop.pmu.models import Bet


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')

class HippodromeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hippodrome
        fields = '__all__'

class PlayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Player
        fields = '__all__'

class RaceSerializer(serializers.ModelSerializer):

    player_set = PlayerSerializer(many=True, read_only=True)

    class Meta:
        model = Race
        fields = '__all__'

class RaceSessionSerializer(serializers.ModelSerializer):
    race_set = RaceSerializer(many=True)

    class Meta:
        model = RaceSession
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

class BetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bet
        fields = '__all__'