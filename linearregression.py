from typing import TypeVar, Literal, cast, Generic

import numpy as np
from numpy import ndarray, dtype, float64, uint8

from lekagemodel import LeakageModel

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
K = TypeVar('K', bound=int)


def expand(labels: ndarray[tuple[B, N, K], dtype[uint8]]) -> ndarray[tuple[Literal[9], B, N, K], dtype[uint8]]:
    return np.concatenate((
        np.unpackbits(labels[np.newaxis, :, :, :], axis=0, bitorder='little'),
        np.ones((1, labels.shape[0], labels.shape[1], labels.shape[2]), dtype=uint8)
    ), axis=0, dtype=uint8)


class Model(Generic[B], LeakageModel[B]):
    coefficients: ndarray[tuple[B, Literal[9]], dtype[float64]]

    def __init__(self, labels: ndarray[tuple[B, N], dtype[uint8]], traces: ndarray[tuple[B, N], dtype[float64]]):
        x: ndarray[tuple[Literal[9], B, N], dtype[uint8]]
        x = expand(cast(
            ndarray[tuple[B, N, Literal[1]], dtype[uint8]],
            labels[:, :, None]
        )).squeeze(axis=3)
        self.coefficients = (np.linalg.inv(
            np.matmul(x.transpose(1, 0, 2), x.transpose(1, 2, 0), dtype=np.int64)
        ) @ x.transpose(1, 0, 2) @ traces[:, :, None]).squeeze(axis=2)

    def loss(
            self,
            traces: ndarray[tuple[B, N], dtype[float64]],
            c:      ndarray[tuple[B, N, K], dtype[uint8]]
    ) -> ndarray[tuple[B, K], dtype[float64]]:

        return np.square(
            traces[:, None, :] - (expand(c).transpose(1, 3, 2, 0) @ self.coefficients[:, None, :, None]).squeeze(3)
        ).sum(axis=2)

    def keys_probability(
            self,
            plaintexts: ndarray[tuple[B, N],    dtype[uint8]],
            traces:     ndarray[tuple[B, N], dtype[float64]],
            keys:       ndarray[tuple[B, K],       dtype[uint8]]
    ) -> ndarray[tuple[B, K], dtype[float64]]:

        return np.exp(-self.loss(traces, keys[:, None, :] ^ plaintexts[:, :, None]))
