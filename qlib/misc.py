import yaml
from types import SimpleNamespace


class DotDict(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, DotDict(value))
            else:
                self.__setattr__(key, value)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def items(self):
        keys = sorted(self.__dict__)
        item_dict = {k: self.__dict__[k] for k in keys}
        return item_dict.items()


def read_config(config_path):
    with open(config_path, "r") as ymlfile:
        return DotDict(yaml.safe_load(ymlfile))
