import datetime

from django.utils.decorators import method_decorator
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from .decorators import cache_page
from django.http import Http404
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly, AllowAny

from django_filters.rest_framework import DjangoFilterBackend

from django.db.models import Count, Avg, Sum

from .serializers import (
    UserTokenSerializer, EmailSerializer, ResetPasswordSerializer,
    FullUserSerializer, UserSerializer,
    ListRaceSerializer, SimpleRaceSerializer, RaceSerializer,
    RaceSessionSerializer,
    PlayerSerializer,
    OddsSerializer,
    HippodromeSerializer, HorseSerializer, OwnerSerializer, HerderSerializer, JockeySerializer, TrainerSerializer,
    BetSerializer
)
from .auth import IsAdmin
from cataclop.users.models import User
from cataclop.core.models import RaceSession, Race, Player, Hippodrome, Horse, Trainer, Herder, Owner, Jockey, Odds
from cataclop.pmu.models import Bet
from cataclop.ml.pipeline import factories
from cataclop.core.signals import user_registered
from cataclop.core import emails

class MultiSerializerViewSetMixin(object):
    def get_serializer_class(self):
        """
        Look for serializer class in self.serializer_action_classes, which
        should be a dict mapping action name (key) to serializer class (value),
        i.e.:

        class MyViewSet(MultiSerializerViewSetMixin, ViewSet):
            serializer_class = MyDefaultSerializer
            serializer_action_classes = {
               'list': MyListSerializer,
               'my_action': MyActionSerializer,
            }

            @action
            def my_action:
                ...

        If there's no entry for that action then just fallback to the regular
        get_serializer_class lookup: self.serializer_class, DefaultSerializer.

        """
        try:
            return self.serializer_action_classes[self.action]
        except (KeyError, AttributeError):
            return super(MultiSerializerViewSetMixin, self).get_serializer_class()


class BaseView(viewsets.ModelViewSet):
    def get_permissions(self):
        if self.action == 'list':
            permission_classes = [AllowAny]
        elif self.action == 'retrieve':
            permission_classes = [IsAuthenticatedOrReadOnly]
        else:
            permission_classes = [IsAdmin]
        return [permission() for permission in permission_classes]


class AuthToken(ObtainAuthToken):

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)

        return Response({
            'token': token.key,
            'user_id': user.pk,
            'email': user.email
        })


class UserViewSet(BaseView):
    queryset = User.objects.all().order_by('-date_joined')

    def get_queryset(self):
        user = self.request.user
        queryset = self.queryset
        if self.action == 'list' and not user.is_staff:
            queryset = queryset.filter(pk=user.pk)
        return queryset

    def get_object(self):
        pk = self.kwargs.get('pk')

        if pk == "me":
            if self.request.user.is_anonymous:
                raise Http404
            return self.request.user

        return super(UserViewSet, self).get_object()

    def perform_create(self, serializer):
        user = serializer.save()
        user_registered.send(sender=self.__class__, id=user.id)

    def get_permissions(self):
        if self.action in ('create', 'verify', 'request_verification', 'request_password_reset', 'reset_password'):
            return [AllowAny()]
        return [IsAuthenticated()]

    def get_serializer_class(self):
        if self.action == 'verify':
            return UserTokenSerializer
        elif self.action in ('request_verification', 'request_password_reset'):
            return EmailSerializer
        elif self.action == 'reset_password':
            return ResetPasswordSerializer
        elif self.request.user.is_staff:
            return FullUserSerializer

        return FullUserSerializer

    @action(detail=False, methods=['post'])
    def verify(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.get(pk=serializer.data.get('uid'), is_active=True)
        user.is_verified = True
        user.save()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'])
    def request_verification(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.get(email=serializer.data.get('email'), is_active=True)
        emails.send_verification_email(user)

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'])
    def request_password_reset(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.get(email=serializer.data.get('email'), is_active=True)
        emails.send_password_reset_request_email(user)

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'])
    def reset_password(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.get(pk=serializer.data.get('uid'), is_active=True)
        user.set_password(serializer.data.get('password'))
        user.save()

        return Response(status=status.HTTP_204_NO_CONTENT)


class RaceSessionViewSet(BaseView):
    queryset = RaceSession.objects.select_related('hippodrome').all()
    serializer_class = RaceSessionSerializer

    permission_classes = [AllowAny]

    filterset_fields = ('date', 'num')

    def get_queryset(self):
        q = self.queryset.order_by('-date', 'num')
        return q

    @method_decorator(cache_page(60))
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)


class RaceViewSet(MultiSerializerViewSetMixin, BaseView):
    queryset = Race.objects.select_related('session', 'session__hippodrome') \
        .prefetch_related('player_set', 'player_set__horse', 'player_set__herder', 'player_set__trainer',
                          'player_set__jockey', 'player_set__owner') \
        .all()
    serializer_class = RaceSerializer

    # serializer_action_classes = {
    #    "list": ListRaceSerializer
    # }

    permission_classes = [AllowAny]

    filterset_fields = ('start_at', 'num')

    def get_queryset(self):
        q = self.queryset.order_by('-start_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(start_at__date=date.date())

        if self.request.query_params.get('horse'):
            q = q.filter(player__horse__id=self.request.query_params.get('horse'))

        if self.request.query_params.get('jockey'):
            q = q.filter(player__jockey__id=self.request.query_params.get('jockey'))

        if self.request.query_params.get('trainer'):
            q = q.filter(player__trainer__id=self.request.query_params.get('trainer'))

        return q

    @method_decorator(cache_page(60))
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)


class PlayerViewSet(BaseView):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('num')
        return q


class OddsViewSet(BaseView):
    queryset = Odds.objects.prefetch_related('player').all()

    serializer_class = OddsSerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('-date')

        if self.request.query_params.get('race'):
            race_id = self.request.query_params.get('race')
            q = q.filter(player__race__id=race_id)

        return q


class HorseViewSet(BaseView):
    queryset = Horse.objects.all()
    serializer_class = HorseSerializer

    def get_queryset(self):
        q = self.queryset.order_by('name')
        if self.request.query_params.get('q'):
            q = q.filter(name__icontains=self.request.query_params.get('q'))
        return q

    @method_decorator(cache_page(60))
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)


class TrainerViewSet(BaseView):
    queryset = Trainer.objects.all()
    serializer_class = TrainerSerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('name')
        if self.request.query_params.get('q'):
            q = q.filter(name__icontains=self.request.query_params.get('q'))
        return q


class JockeyViewSet(BaseView):
    queryset = Jockey.objects.all()
    serializer_class = JockeySerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('name')
        if self.request.query_params.get('q'):
            q = q.filter(name__icontains=self.request.query_params.get('q'))
        return q


class OwnerViewSet(BaseView):
    queryset = Owner.objects.all()
    serializer_class = OwnerSerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('name')
        if self.request.query_params.get('q'):
            q = q.filter(name__icontains=self.request.query_params.get('q'))
        return q


class HerderViewSet(BaseView):
    queryset = Herder.objects.all()
    serializer_class = HerderSerializer

    permission_classes = [AllowAny]

    def get_queryset(self):
        q = self.queryset.order_by('name')
        if self.request.query_params.get('q'):
            q = q.filter(name__icontains=self.request.query_params.get('q'))
        return q


class BetViewSet(BaseView):
    queryset = Bet.objects.all()
    serializer_class = BetSerializer

    permission_classes = [IsAdmin]

    def get_queryset(self):
        q = self.queryset.order_by('-created_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(created_at__date=date.date())

        return q


@api_view(['post'])
@permission_classes([IsAdmin])
def post_live_odds(request):
    now = datetime.datetime.now()
    date = request.data.get('date', now.strftime('%Y-%m-%d'))

    R = request.data.get('R')
    C = request.data.get('C')

    try:
        race = Race.objects.get(start_at__date=date, session__num=R, num=C)
    except ObjectDoesNotExist:
        return Response(None, status=status.HTTP_404_NOT_FOUND)

    for odds in request.data.get('odds'):
        offline = odds.get('offline', False)
        ts = odds.get('ts')
        value = odds.get('value')
        evolution = odds.get('evolution', 0)
        player = race.get_player(odds.get('player'))
        whale = odds.get('whale', False)

        if player is None:
            continue

        odds = Odds(value=value, is_final=False, is_final_ref=False)
        odds.evolution = evolution
        odds.date = datetime.datetime.fromtimestamp(ts)
        odds.player = player
        odds.whale = whale
        odds.offline = offline

        odds.save()

    return Response()


@api_view(['get'])
@permission_classes([IsAdmin])
def predict(request):
    now = datetime.datetime.now()
    date = request.query_params.get('date', now.strftime('%Y-%m-%d'))
    R = request.query_params.get('R')
    C = request.query_params.get('C')
    p = request.query_params.get('program', '2020-05-25')

    race = Race.objects.get(start_at__date=date, session__num=R, num=C)
    program = factories.Program.factory(p)

    program.predict(dataset_params={
        'race_id': race.id
    }, locked=True, dataset_reload=True)

    bets = program.bet()

    if bets:
        return Response(bets.to_dict(orient='records'))
    return Response()


@api_view(['get'])
@permission_classes([IsAdmin])
def stats(request, pk=None):
    stats = Bet.objects.filter(simulation=False, player__isnull=False).values('program') \
        .annotate(pcount=Count('program'), win=Sum('player__winner_dividend'))
    return Response(stats)
