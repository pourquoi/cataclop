import urllib.parse
from django.template.loader import render_to_string
from django.core.mail import EmailMessage
from django.urls import reverse
from .auth import make_user_token
from .utils import get_absolute_url
from cataclop.settings import (
    EMAIL_RECIPIENTS, PROJECT_NAME
)


def send_verification_email(user):
    token = make_user_token(user)

    message = render_to_string("emails/user_verification_request.mjml.html", {
        'user': user,
        'token': token,
        'cta_url': get_absolute_url(reverse('user-verify') + '?' + 'uid=' + str(user.id) + '&token=' + urllib.parse.quote(token))
    })

    msg = EmailMessage(
        subject=f"{PROJECT_NAME} - Confirm your email",
        body=message,
        to=EMAIL_RECIPIENTS or (user.email,)
    )
    msg.content_subtype = "html"
    msg.send()


def send_password_reset_request_email(user):
    token = make_user_token(user)

    message = render_to_string("emails/password_reset_request.mjml.html", {
        'user': user,
        'token': token,
        'cta_url': get_absolute_url(reverse('user-reset-password') + '?' + 'uid=' + str(user.id) + '&token=' + urllib.parse.quote(token))
    })

    msg = EmailMessage(
        subject=f"{PROJECT_NAME} - Change your password",
        body=message,
        to=EMAIL_RECIPIENTS or (user.email,)
    )
    msg.content_subtype = "html"
    msg.send()
