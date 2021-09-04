from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from django.conf.urls.static import static

from rest_framework import routers
from cataclop.api import urls as api_urls
from cataclop.front import urls as front_urls

from django.conf import settings

urlpatterns = [
    url(r'^api/auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^api/', include(api_urls)),
    path('admin/', admin.site.urls),
    url('', include(front_urls))
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        path('__debug__', include(debug_toolbar.urls))
    ] + urlpatterns