"""Factory for models. 
The model should be registered by the decorator @register_model(name). 
The dataset can be retrieved by get_model(name, **kwargs). 
The supported datasets can be retrieved by get_model_names().
"""
import warnings

__MODEL__ = {}
def register_model(name):
    def wrapper(cls):   
        if __MODEL__.get(name, None):
            if __MODEL__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __MODEL__[name] = cls
        cls.name = name
        return cls
    return wrapper

def get_model(name: str, **kwargs):
    if __MODEL__.get(name, None) is None:
        raise NameError(f"Model '{name}' is not defined.")
    return __MODEL__[name](**kwargs)

def get_model_names():
    return list(__MODEL__.keys())

