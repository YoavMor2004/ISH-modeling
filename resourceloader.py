import json
from typing import Any, Optional, Self, TypedDict, cast, Type, TypeVar

from numpy import ndarray, dtype, uint8, float64
from scipy.io import loadmat  # type: ignore


class Profile(TypedDict):
    labels: ndarray[tuple[int, int], dtype[uint8]]
    traces: ndarray[tuple[int, int, int], dtype[float64]]


class Attack(TypedDict):
    plaintexts: ndarray[tuple[int, int], dtype[uint8]]
    traces: ndarray[tuple[int, int, int], dtype[float64]]


T = TypeVar('T', Profile, Attack)


def profile_new(d: dict[str, ndarray[tuple, dtype[Any]]], /) -> Optional[Profile]:
    return verify(d, Profile, 'labels', 'traces')


def attack_new(d: dict[str, ndarray[tuple, dtype[Any]]], /) -> Optional[Attack]:
    return verify(d, Attack, 'plaintexts', 'traces')


def verify(
        d: dict[str, ndarray[tuple, dtype[Any]]],
        /,
        profile_type: Type[T],
        labels_name: str,
        traces_name: str
) -> Optional[T]:
    if labels_name not in d:
        return None
    if traces_name not in d:
        return None
    labels = d[labels_name]
    traces = d[traces_name]
    del d
    if labels.ndim < 2:
        return None
    if traces.ndim < 3:
        return None
    labels = labels.reshape(labels.shape[:2])
    traces = traces.reshape(traces.shape[:3])
    if labels.shape != traces.shape[0:2]:
        return None
    if labels.dtype != uint8:
        return None
    if traces.dtype != float64:
        return None
    labels = cast(ndarray[tuple[int, int],      dtype[uint8]],   labels)
    traces = cast(ndarray[tuple[int, int, int], dtype[float64]], traces)
    return profile_type({labels_name: labels, traces_name: traces})  # type: ignore


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

    def load(self, file_name: str, profile_type: Type[T]) -> Optional[T]:
        if file_name not in self:
            return None
        if profile_type is Profile:
            return profile_new(loadmat(self[file_name]))
        if profile_type is Attack:
            return attack_new(loadmat(self[file_name]))
        else:
            return None
