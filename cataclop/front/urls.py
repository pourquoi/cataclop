from django.urls import path

from cataclop.front.views import home

urlpatterns = [
    path('', home, name='home')
]