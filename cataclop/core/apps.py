from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'cataclop.core'
    verbose_name = 'Core Application'

    def ready(self):
        import cataclop.core.signals