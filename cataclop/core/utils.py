import logging
from cataclop.settings import HOST, SCHEME, PORT
logger = logging.getLogger(__name__)


def get_absolute_url(path):
    url = SCHEME + '://' + HOST
    if PORT is not None and ((SCHEME == 'http' and PORT != 80) or (SCHEME == 'https' and PORT != 443)):
        url = url + ':' + str(PORT)
    return url + path