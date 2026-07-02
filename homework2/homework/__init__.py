from .models import load_model as load_model

__all__ = ["load_model"]


def __getattr__(name):
    if name == "logger":
        import importlib

        return importlib.import_module(".logger", __name__)
    raise AttributeError(name)
