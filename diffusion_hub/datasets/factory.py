"""
"""
import warnings

__DATASET__ = {}
def register_dataset(name):
    def wrapper(cls):   
        if __DATASET__.get(name, None):
            if __DATASET__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DATASET__[name] = cls
        cls.name = name
        return cls
    return wrapper

def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset '{name}' is not defined.")
    return __DATASET__[name](**kwargs)

def get_supported_datasets():
    return list(__DATASET__.keys())