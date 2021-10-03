from django.core import mail
from unittest.mock import patch
from rest_framework.test import APITestCase
from rest_framework import status
from cataclop.users.tests.utils import factories as user_factories
from cataclop.users.models import User
from cataclop.core.auth import make_user_token


class UserApiTest(APITestCase):

    def test_login(self):
        user = user_factories.UserFactory.create(email="bob@example.com")
        user.save()
        response = self.client.post('/api/token/', {'email': "bob@example.com", 'password': "pass1234"}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        token = response.data.get('token')
        self.client.credentials(HTTP_AUTHORIZATION='Bearer ' + token)
        response = self.client.get('/api/users/me/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_register(self):
        with patch('cataclop.core.emails.make_user_token') as mock_make_user_token:
            mock_make_user_token.return_value = 'generated_token'

            response = self.client.post('/api/users/', {'email': "bob@example.com", 'password': "pass1234"}, format='json')
            self.assertEqual(response.status_code, status.HTTP_201_CREATED)

            user = User.objects.get(email="bob@example.com")
            self.assertFalse(user.is_verified)

            self.assertEqual(len(mail.outbox), 1)
            self.assertTrue('generated_token' in mail.outbox[0].body)

    def test_verify(self):
        user = user_factories.UserFactory.create(email="bob@example.com", is_verified=False)
        user.save()
        token = make_user_token(user)
        response = self.client.post('/api/users/verify/', {'uid': user.id, 'token': token}, format='json')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        user.refresh_from_db()
        self.assertTrue(user.is_verified)

    def test_request_verification(self):
        user = user_factories.UserFactory.create(email="bob@example.com", is_verified=False)
        user.save()

        with patch('cataclop.core.emails.make_user_token') as mock_make_user_token:
            mock_make_user_token.return_value = 'generated_token'
            response = self.client.post('/api/users/request_verification/', {'email': "bob@example.com"}, format='json')
            self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
            self.assertEqual(len(mail.outbox), 1)
            self.assertTrue('generated_token' in mail.outbox[0].body)

    def test_reset_password(self):
        user = user_factories.UserFactory.create(email="bob@example.com", is_verified=False)
        user.save()
        token = make_user_token(user)
        response = self.client.post('/api/users/reset_password/', {'uid': user.id, 'token': token, 'password': "pass4567"}, format='json')
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        user.refresh_from_db()
        self.assertTrue(user.check_password("pass4567"))

    def test_request_password_reset(self):
        user = user_factories.UserFactory.create(email="bob@example.com")
        user.save()

        with patch('cataclop.core.emails.make_user_token') as mock_make_user_token:
            mock_make_user_token.return_value = 'generated_token'
            response = self.client.post('/api/users/request_password_reset/', {'email': "bob@example.com"}, format='json')
            self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
            self.assertEqual(len(mail.outbox), 1)
            self.assertTrue('generated_token' in mail.outbox[0].body)

    def test_get_users(self):
        user = user_factories.UserFactory.create(email="bob@example.com")
        user.save()
        self.client.force_authenticate(user=user)
        response = self.client.get('/api/users/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data.get('count'), 1)

    def test_get_me(self):
        user = user_factories.UserFactory.create(email="bob@example.com")
        user.save()
        self.client.force_authenticate(user=user)
        response = self.client.get('/api/users/me/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data.get('email'), user.email)

