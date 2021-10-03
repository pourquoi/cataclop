from rest_framework import serializers

from django.core.exceptions import ValidationError as DjangoValidationError
from django.contrib.auth.password_validation import validate_password
from rest_framework.serializers import ValidationError
from cataclop.users.models import User
from cataclop.core.models import RaceSession, Race, Player, Odds, Hippodrome, Horse, Trainer, Herder, Owner, Jockey
from cataclop.pmu.models import Bet
from cataclop.core.auth import get_user_from_token


class EmailSerializer(serializers.Serializer):
    email = serializers.EmailField()


class UserTokenSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()

    def validate(self, attrs):
        validated_data = super().validate(attrs)
        user = get_user_from_token(self.initial_data.get('uid'), self.initial_data.get('token'))
        if user is None:
            raise ValidationError()
        return validated_data


class ResetPasswordSerializer(serializers.Serializer):
    token = serializers.CharField()
    uid = serializers.CharField()
    password = serializers.CharField(style={'input_type': 'password'})

    def validate(self, attrs):
        validated_data = super().validate(attrs)
        user = get_user_from_token(self.initial_data.get('uid'), self.initial_data.get('token'))

        try:
            validate_password(attrs['password'], user)
        except DjangoValidationError as e:
            raise serializers.ValidationError(detail="test")
        return attrs


class FullUserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(style={'input_type': 'password'}, write_only=True)

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            is_active=True,
            is_verified=False
        )
        return user

    def validate(self, attrs):
        user = User(**attrs)
        password = attrs.get('password')
        try:
            validate_password(password, user)
        except DjangoValidationError as e:
            raise serializers.ValidationError()
        return attrs

    class Meta:
        model = User
        fields = ('id', 'url', 'username', 'email', 'groups', 'is_staff', 'is_superuser', 'date_joined', 'last_login', 'password', 'is_active')


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


class SimplePlayerSerialize(serializers.ModelSerializer):
    horse = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Player
        fields = ('id', 'num', 'horse')


class OddsSerializer(serializers.ModelSerializer):
    player = SimplePlayerSerialize(read_only=True)

    class Meta:
        model = Odds
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
