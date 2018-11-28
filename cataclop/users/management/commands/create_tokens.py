from cataclop.users.models import User
from rest_framework.authtoken.models import Token
from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):

    def handle(self, *args, **options):

        for user in User.objects.all():
            Token.objects.get_or_create(user=user)
        