from django.urls import path

from cataclop.front.views import dashboard

urlpatterns = [
    path('', dashboard, name='dashboard')
]