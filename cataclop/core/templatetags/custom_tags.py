from django import template
from urllib.parse import urljoin
from cataclop.settings import STATIC_URL
from cataclop.core.utils import get_absolute_url

register = template.Library()


@register.simple_tag
def static_abs(path):
    return get_absolute_url(urljoin(STATIC_URL, path))
