from rest_framework import serializers

from cataclop.users.models import User
from cataclop.core.models import Race
from cataclop.pmu.models import Bet


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')

class RaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Race
        fields = '__all__'

class BetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bet
        fields = '__all__'