from django.contrib.auth.backends import ModelBackend


class AppModelBackend(ModelBackend):

    def user_can_authenticate(self, user):
        is_verified = getattr(user, 'is_verified', None)
        if not user.is_superuser and is_verified is False:
            return False

        is_active = getattr(user, 'is_active', None)
        return is_active or is_active is None
