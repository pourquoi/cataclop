import datetime
from .serializers import UserSerializer, RaceSerializer, BetSerializer
from rest_framework import viewsets
from rest_framework.decorators import action, api_view
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

from django.db.models import Count, Avg, Sum

from cataclop.users.models import User
from cataclop.core.models import Race
from cataclop.pmu.models import Bet
from cataclop.ml.pipeline import factories

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

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer

class RaceViewSet(viewsets.ModelViewSet):
    queryset = Race.objects.all()
    serializer_class = RaceSerializer

    def get_queryset(self):

        q = self.queryset.order_by('-start_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(start_at__date = date.date())

        return q

class BetViewSet(viewsets.ModelViewSet):
    queryset = Bet.objects.all()
    serializer_class = BetSerializer

    @action(methods=['get'], detail=False)
    def stats(self, request, pk=None):
        stats = Bet.objects.filter(simulation=False, player__isnull=False).values('program')\
            .annotate(pcount=Count('program'), win=Sum('player__winner_dividend'))
        return Response(stats)

    def get_queryset(self):
        q = self.queryset.order_by('-created_at')

        if self.request.query_params.get('date'):
            date = datetime.datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
            q = q.filter(created_at__date = date.date())

        return q


@api_view(['get'])
def predict(request):
    now = datetime.datetime.now()
    date = request.query_params.get('date', now.strftime('%Y-%m-%d'))
    R = request.query_params.get('R')
    C = request.query_params.get('C')
    p = request.query_params.get('program', '2020-05-25')

    race = Race.objects.get(start_at__date=date, session__num=R, num=C)
    program = factories.Program.factory(p)

    print(race)

    program.predict(dataset_params = {
        'race_id': race.id
    }, locked=True, dataset_reload=True)

    bets = program.bet()
    print(bets)
    return Response(bets.to_dict(orient='records'))
