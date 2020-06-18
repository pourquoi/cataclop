from django.views.decorators.cache import decorator_from_middleware_with_args
from django.middleware.cache import CacheMiddleware


class LocalCacheMiddleware(CacheMiddleware):
    def process_request(self, request):
        response = super(LocalCacheMiddleware, self).process_request(request)
        # Add X-Cache: HIT header if response is returned from cache
        if response:
            response['X-Cache'] = 'HIT'
        return response

    def process_response(self, request, response):
        response = super(LocalCacheMiddleware, self).process_response(request, response)
        # Add X-Cache: MISS header since we missed the cache
        response['X-Cache'] = 'MISS'
        return response


def cache_page(*args, **kwargs):
    """
    c/p cache_page decorator from django.views.decorators.cache and change
    middleware from CacheMiddleware to LocalCacheMiddleware
    """
    if len(args) != 1 or callable(args[0]):
        raise TypeError("cache_page has a single mandatory positional argument: timeout")
    cache_timeout = args[0]
    cache_alias = kwargs.pop('cache', None)
    key_prefix = kwargs.pop('key_prefix', None)
    if kwargs:
        raise TypeError("cache_page has two optional keyword arguments: cache and key_prefix")

    return decorator_from_middleware_with_args(LocalCacheMiddleware)(
        cache_timeout=cache_timeout, cache_alias=cache_alias, key_prefix=key_prefix
    )