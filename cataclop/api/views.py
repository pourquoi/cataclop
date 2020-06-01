import datetime

from rest_framework import viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django_filters.rest_framework import DjangoFilterBackend

from django.db.models import Count, Avg, Sum

from .serializers import (
    UserSerializer, RaceSerializer, BetSerializer, RaceSessionSerializer,
    HippodromeSerializer, HorseSerializer, OwnerSerializer, HerderSerializer, JockeySerializer,
    TrainerSerializer, PlayerSerializer
)
from .auth import IsAdmin
from cataclop.users.models import User
from cataclop.core.models import RaceSession, Race, Player, Hippodrome, Horse, Trainer, Herder, Owner, Jockey
from cataclop.pmu.models import Bet
from cataclop.ml.pipeline import factories


class BaseView(viewsets.ModelViewSet):
    def get_permissions(self):
        if self.action == 'list':
            permission_classes = [IsAuthenticated]
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
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

    def get_object(self):
        pk = self.kwargs.get('pk')

        if pk == "current":
            return self.request.user

        return super(UserViewSet, self).get_object()

class RaceSessionViewSet(BaseView):
    queryset = RaceSession.objects.all()
    serializer_class = RaceSessionSerializer

    def get_queryset(self):
        q = self.queryset.order_by('-date', 'num')
        return q

class RaceViewSet(BaseView):
    queryset = Race.objects.all()
    serializer_class = RaceSerializer

    def get_queryset(self):
        q = self.queryset.order_by('-start_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(start_at__date = date.date())

        return q

class PlayerViewSet(BaseView):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer

    def get_queryset(self):
        q = self.queryset.order_by('num')

class HorseViewSet(BaseView):
    queryset = Horse.objects.all()
    serializer_class = HorseSerializer

class TrainerViewSet(BaseView):
    queryset = Trainer.objects.all()
    serializer_class = TrainerSerializer

class JockeyViewSet(BaseView):
    queryset = Jockey.objects.all()
    serializer_class = JockeySerializer

class OwnerViewSet(BaseView):
    queryset = Owner.objects.all()
    serializer_class = OwnerSerializer

class HerderViewSet(BaseView):
    queryset = Herder.objects.all()
    serializer_class = HerderSerializer

class BetViewSet(BaseView):
    queryset = Bet.objects.all()
    serializer_class = BetSerializer

    def get_queryset(self):
        q = self.queryset.order_by('-created_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(created_at__date = date.date())

        return q

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

    program.predict(dataset_params = {
        'race_id': race.id
    }, locked=True, dataset_reload=True)

    bets = program.bet()

    if bets:
        return Response(bets.to_dict(orient='records'))
    return Response()


@api_view(['get'])
def stats(self, request, pk=None):
    stats = Bet.objects.filter(simulation=False, player__isnull=False).values('program')\
        .annotate(pcount=Count('program'), win=Sum('player__winner_dividend'))
    return Response(stats)
