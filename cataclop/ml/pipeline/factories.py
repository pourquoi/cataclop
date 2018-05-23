import json
import hashlib
import importlib
import os

from abc import ABC, abstractmethod
from cataclop.ml.settings import DATA_DIR, MODEL_DIR, PROGRAM_DIR

class PipelineEntity(ABC):

    def __init__(self, name, params=None, version=None):
        self.name = name

        self.version = '1.0'

        if version is not None:
            self.version = version

        self.params = self.defaults.copy()

        if params is not None:
            self.params.update(params)

    @classmethod
    def factory(cls, name, params=None):
        module = importlib.import_module(cls.get_class_path() + '.' + name)
        concrete_cls = getattr(module, cls.get_class_name(), None)
        return concrete_cls(name, params)

    @classmethod
    @abstractmethod
    def get_class_path(cls):
        pass

    @classmethod
    @abstractmethod
    def get_class_name(cls):
        pass

    @classmethod
    @abstractmethod
    def get_data_base_dir(cls):
        pass

    @property
    @abstractmethod
    def defaults(self):
        pass

    @property
    def hash(self):
        s = self.name + self.version + json.dumps(self.params, sort_keys=True)
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    @property
    def data_dir(self):
        return os.path.join(self.get_data_base_dir(), self.name + '-' + self.version + '-' + self.hash)



class Program(PipelineEntity):

    @classmethod
    def get_class_path(cls):
        return 'cataclop.ml.pipeline.programs'

    @classmethod
    def get_class_name(cls):
        return 'Program'

    @classmethod
    def get_data_base_dir(cls):
        return PROGRAM_DIR

    def run(self, **kwargs):
        pass


class Dataset(PipelineEntity):

    @classmethod
    def get_class_path(cls):
        return 'cataclop.ml.pipeline.datasets'

    @classmethod
    def get_class_name(cls):
        return 'Dataset'

    @classmethod
    def get_data_base_dir(cls):
        return DATA_DIR

    def load(self, force=False):
        pass

    def save(self, clear=False):
        pass


class Model(PipelineEntity):

    @classmethod
    def get_class_path(cls):
        return 'cataclop.ml.pipeline.models'

    @classmethod
    def get_class_name(cls):
        return 'Model'

    @classmethod
    def get_data_base_dir(cls):
        return MODEL_DIR

    def load(self):
        pass

    def save(self):
        pass