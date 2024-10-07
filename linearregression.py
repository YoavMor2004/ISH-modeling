from typing import TypeVar, Literal, cast, Generic

import numpy as np
from numpy import ndarray, dtype, float64, uint8

from lekagemodel import LeakageModel

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
K = TypeVar('K', bound=int)


def expand(labels: ndarray[tuple[B, N, K], dtype[uint8]]) -> ndarray[tuple[Literal[9], B, N, K], dtype[uint8]]:
    return np.concatenate((
        np.unpackbits(labels[np.newaxis, :, :, :], axis=0, bitorder='little'),
        np.ones((1, labels.shape[0], labels.shape[1], labels.shape[2]), dtype=uint8)
    ), axis=0, dtype=uint8)


class Model(Generic[P], LeakageModel[P]):
    coefficients: ndarray[tuple[Literal[9], P], dtype[float64]]

    def __init__(self, labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N, P], dtype[float64]]):
        x: ndarray[tuple[Literal[9], N], dtype[uint8]]
        x = expand(cast(
            ndarray[tuple[Literal[1], N, Literal[1]], dtype[uint8]],
            labels[None, :, None]
        )).squeeze(axis=(1, 3))
        self.coefficients = np.linalg.inv(np.matmul(x, x.T, dtype=np.int64)) @ x @ traces

    def loss(
            self,
            traces: ndarray[tuple[B, N, P], dtype[float64]],
            c:      ndarray[tuple[B, N, K], dtype[uint8]]
    ) -> ndarray[tuple[B, K, P], dtype[float64]]:

        return np.square(traces[:, None, :, :] - expand(c).transpose(1, 3, 2, 0) @ self.coefficients).sum(axis=2)

    def keys_probability(
            self,
            plaintexts: ndarray[tuple[B, N],    dtype[uint8]],
            traces:     ndarray[tuple[B, N, P], dtype[float64]],
            keys:       ndarray[tuple[B, K],       dtype[uint8]]
    ) -> ndarray[tuple[B, K, P], dtype[float64]]:

        return np.exp(-self.loss(traces, keys[:, None, :] ^ plaintexts[:, :, None]))
