from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar, Generic, Literal, cast, reveal_type

import numpy as np
from numpy import ndarray, dtype, uint8, float64

N = TypeVar('N', bound=int)


@dataclass
class TemplateModel:
    mean_of_classes: ndarray[tuple[Literal[256]], dtype[float64]]
    std_of_classes: ndarray[tuple[Literal[256]], dtype[float64]]


def model(labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N], dtype[float64]]) -> TemplateModel:
    mean_of_classes: ndarray[tuple[Literal[256]], dtype[float64]]
    std_of_classes: ndarray[tuple[Literal[256]], dtype[float64]]

    mean_of_classes = np.empty(256, dtype=float64)
    std_of_classes = np.empty(256, dtype=float64)
    for x in range(256):
        c: ndarray[tuple[int], dtype[float64]]
        c = traces[labels == x]
        mean_of_classes[x] = c.mean()
        std_of_classes[x] = c.std()
    return TemplateModel(mean_of_classes, std_of_classes)
