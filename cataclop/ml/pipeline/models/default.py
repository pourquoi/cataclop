from cataclop.ml.pipeline import factories

class Model(factories.Model):

    def __init__(self, name, params):
        super().__init__(name, params)

    @property
    def defaults(self):
        return {

        }

    def load(self):
        pass

    def save(self):
        pass