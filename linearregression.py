from typing import TypeVar, Literal, cast, Generic

import numpy as np
from numpy import ndarray, dtype, float64, uint8

from lekagemodel import LeakageModel

N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
K = TypeVar('K', bound=int)


def expand(labels: ndarray[tuple[N, K], dtype[uint8]]) -> ndarray[tuple[Literal[9], N, K], dtype[uint8]]:
    x: ndarray[tuple[Literal[9], N, K], dtype[uint8]]
    x = np.concatenate((
        np.unpackbits(labels[np.newaxis, :, :], axis=0, bitorder='little'),
        np.ones((1, labels.shape[0], labels.shape[1]), dtype=uint8)
    ), axis=0, dtype=uint8)
    return x


class Model(Generic[P], LeakageModel[P]):
    coefficients: ndarray[tuple[Literal[9], P], dtype[float64]]

    def __init__(self, labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N, P], dtype[float64]]):
        x: ndarray[tuple[Literal[9], N], dtype[uint8]]
        x = expand(cast(
            ndarray[tuple[N, Literal[1]], dtype[uint8]],
            labels[:, None]
        )).squeeze(axis=2)
        self.coefficients = np.linalg.inv(np.matmul(x, x.T, dtype=np.int64)) @ x @ traces

    def loss(
            self,
            traces: ndarray[tuple[N, P], dtype[float64]],
            c:      ndarray[tuple[N, K], dtype[uint8]]
    ) -> ndarray[tuple[K, P], dtype[float64]]:

        return np.square(traces[None, :, :] - expand(c).transpose(2, 1, 0) @ self.coefficients).sum(axis=1)

    def keys_probability(
            self,
            plaintexts: ndarray[tuple[N],    dtype[uint8]],
            traces:     ndarray[tuple[N, P], dtype[float64]],
            keys:       ndarray[tuple[K],    dtype[uint8]]
    ) -> ndarray[tuple[K, P], dtype[float64]]:

        temp = self.loss(traces, keys[None, :] ^ plaintexts[:, None])
        return temp / temp.sum(axis=0, keepdims=True)
