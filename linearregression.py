from typing import TypeVar, Literal

import numpy as np
from numpy import ndarray, dtype, float64, uint8

N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)


def model(labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N, P], dtype[float64]])\
        -> ndarray[tuple[Literal[9], P], dtype[float64]]:

    x: ndarray[tuple[Literal[9], N], dtype[uint8]]
    x = np.concatenate((
        np.unpackbits(labels[np.newaxis, :], axis=0, bitorder='little'),
        np.ones((1, labels.size))
    ), axis=0, dtype=uint8)

    return np.linalg.inv(x @ x.T) @ x @ traces
