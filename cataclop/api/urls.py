from rest_framework.authtoken import views as auth_views
from rest_framework import routers
from django.conf.urls import url, include

from cataclop.api import views as api_views

router = routers.DefaultRouter()
router.register(r'users', api_views.UserViewSet)
router.register(r'races', api_views.RaceViewSet)

urlpatterns = [
    url(r'^login', api_views.AuthToken.as_view())
]

urlpatterns += router.urls