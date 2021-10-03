from django.contrib.auth.tokens import default_token_generator
from cataclop.users.models import User


def make_user_token(user):
    return default_token_generator.make_token(user)


def get_user_from_token(uid, token):
    user = User.objects.get(pk=uid)
    return user if default_token_generator.check_token(user, token) else None

