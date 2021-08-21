from rest_framework.pagination import LimitOffsetPagination


class ApiPagination(LimitOffsetPagination):
    max_limit = 1000
