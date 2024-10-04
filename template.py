from typing import TypeVar, Generic, Literal, cast

import numpy as np
from numpy import ndarray, dtype, uint8, float64

from lekagemodel import LeakageModel

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
K = TypeVar('K', bound=int)


class Model(Generic[P], LeakageModel[P]):
    mean_of_classes: ndarray[tuple[Literal[256], P], dtype[float64]]
    std_of_classes: ndarray[tuple[Literal[256], P], dtype[float64]]

    def __init__(self, labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N, P], dtype[float64]]):
        self.mean_of_classes = np.empty((256, traces.shape[1]), dtype=float64)
        self.std_of_classes = np.empty((256, traces.shape[1]), dtype=float64)
        for x in range(256):
            c: ndarray[tuple[int, P], dtype[float64]]
            c = traces[labels == x]
            self.mean_of_classes[x] = c.mean(axis=0)
            self.std_of_classes[x] = c.std(axis=0)

    def keys_probability(
            self,
            plaintexts: ndarray[tuple[B, N],    dtype[uint8]],
            traces:     ndarray[tuple[B, N, P], dtype[float64]],
            keys:       ndarray[tuple[K],       dtype[uint8]]
    ) -> ndarray[tuple[B, K, P], dtype[float64]]:

        c = cast(ndarray[tuple[B, N, K], dtype[np.int64]], keys[None, None, :] ^ plaintexts[:, :, None])
        temp = Model[P].match(
            traces,
            cast(ndarray[tuple[B, N, K, P], dtype[float64]], self.mean_of_classes[c, :]),
            cast(ndarray[tuple[B, N, K, P], dtype[float64]], self.std_of_classes[c, :])
        )
        return temp / temp.sum(axis=1, keepdims=True)

    @staticmethod
    def match(
            traces:          ndarray[tuple[B, N, P],    dtype[float64]],
            mean_of_classes: ndarray[tuple[B, N, K, P], dtype[float64]],
            std_of_classes:  ndarray[tuple[B, N, K, P], dtype[float64]]
    ) -> ndarray[tuple[B, K, P], dtype[float64]]:

        return np.log(
            1 / (np.sqrt(2 * np.pi) * std_of_classes) * np.exp(-(traces[:, :, np.newaxis, :] - mean_of_classes) ** 2)
        ).sum(axis=0)
