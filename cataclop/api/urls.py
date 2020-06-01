from rest_framework.authtoken import views as auth_views
from rest_framework_simplejwt import views as jwt_views
from rest_framework import routers
from django.conf.urls import url, include

from cataclop.api import views as api_views

router = routers.DefaultRouter()
router.register(r'users', api_views.UserViewSet)
router.register(r'races', api_views.RaceViewSet)
router.register(r'sessions', api_views.RaceSessionViewSet)
router.register(r'players', api_views.PlayerViewSet)
router.register(r'horses', api_views.HorseViewSet)
router.register(r'trainers', api_views.TrainerViewSet)
router.register(r'owners', api_views.OwnerViewSet)
router.register(r'herders', api_views.HerderViewSet)
router.register(r'jockeys', api_views.JockeyViewSet)
router.register(r'bets', api_views.BetViewSet)

urlpatterns = [
    url(r'^predict', api_views.predict),

    url(r'^token/', jwt_views.TokenObtainSlidingView.as_view(), name='token_obtain_pair'),
    url(r'^token/refresh/', jwt_views.TokenRefreshSlidingView.as_view(), name='token_refresh'),
]

urlpatterns += router.urls