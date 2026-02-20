import importlib as _importlib
import typing as _typing

from . import samplers, schedulers, trainers

__all__ = ["eval", "samplers", "schedulers", "trainers"]


def __getattr__(name: str) -> _typing.Any:
    if name == "eval":
        return _importlib.import_module(".eval", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
