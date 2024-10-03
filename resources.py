import json
from typing import Any, Optional, cast

from numpy import ndarray, dtype
from scipy.io import loadmat  # type: ignore


def new(file_path: str) -> Optional[dict[str, str]]:
    resources: Any
    try:
        with open(file_path, 'r') as file:
            resources = json.load(file)
    except FileNotFoundError:
        return None

    if not isinstance(resources, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in resources.items()):
        return None

    return cast(dict[str, str], resources)


def load(resource_loader: dict[str, str], file_name: str) -> Optional[dict[str, ndarray[Any, dtype[Any]]]]:
    if file_name not in resource_loader:
        return None
    return {
        k: v for k, v in loadmat(resource_loader[file_name]).items() if isinstance(k, str) and isinstance(v, ndarray)
    }
