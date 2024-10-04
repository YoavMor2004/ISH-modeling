import json
from typing import Any, Optional, Self

from numpy import ndarray, dtype
from scipy.io import loadmat  # type: ignore


class Resources(dict[str, str]):
    def __init__(self, data: dict[str, str]):
        super().__init__(data)

    @classmethod
    def new(cls, file_path: str) -> Optional[Self]:
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

        return cls(resources)

    def load(self, file_name: str) -> Optional[dict[str, ndarray[tuple, dtype]]]:
        if file_name not in self:
            return None
        return {
            k: v for k, v in loadmat(self[file_name]).items() if isinstance(k, str) and isinstance(v, ndarray)
        }
