import factory
from faker import Faker

fake = Faker()


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'users.User'

    first_name = "Bobby"
    last_name = "Fischer"
    email = factory.Sequence(lambda n: 'user%d@example.com' % n)
    password = factory.PostGenerationMethodCall('set_password', 'pass1234')
    is_verified = True
