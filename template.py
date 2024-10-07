from typing import TypeVar, Generic, Literal, cast

import numpy as np
from numpy import ndarray, dtype, uint8, float64

from lekagemodel import LeakageModel

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
K = TypeVar('K', bound=int)


class Model(Generic[B], LeakageModel[B]):
    mean_of_classes: ndarray[tuple[B, Literal[256]], dtype[float64]]
    std_of_classes:  ndarray[tuple[B, Literal[256]], dtype[float64]]

    def __init__(self, labels: ndarray[tuple[B, N], dtype[uint8]], traces: ndarray[tuple[B, N], dtype[float64]]):
        self.mean_of_classes = np.empty((traces.shape[0], 256), dtype=float64)
        self.std_of_classes = np.empty((traces.shape[0], 256), dtype=float64)
        for b in range(16):
            for x in range(256):
                block_class_traces = cast(ndarray[tuple[int], dtype[float64]], traces[b, labels[b, :] == x])
                self.mean_of_classes[b, x] = block_class_traces.mean()
                self.std_of_classes[b, x] = block_class_traces.std()

    def keys_probability(
            self,
            plaintexts: ndarray[tuple[B, N], dtype[uint8]],
            traces:     ndarray[tuple[B, N], dtype[float64]],
            keys:       ndarray[tuple[B, K], dtype[uint8]]
    ) -> ndarray[tuple[B, K], dtype[float64]]:

        c = cast(ndarray[tuple[B, N, K], dtype[np.uint8]], keys[:, None, :] ^ plaintexts[:, :, None])
        return Model[B].match(
            traces,
            self.mean_of_classes[np.arange(keys.shape[0])[:, None, None], c],
            self.std_of_classes[np.arange(keys.shape[0])[:, None, None], c]
        )

    @staticmethod
    def match(
            traces:          ndarray[tuple[B, N],    dtype[float64]],
            mean_of_classes: ndarray[tuple[B, N, K], dtype[float64]],
            std_of_classes:  ndarray[tuple[B, N, K], dtype[float64]]
    ) -> ndarray[tuple[B, K], dtype[float64]]:

        norm: ndarray[tuple[B, K], dtype[float64]]
        norm = np.sum((traces[:, :, None] - mean_of_classes) ** 2 / std_of_classes ** 2, axis=1)
        return -np.log(2 * np.pi) * traces.shape[1] / 2 - np.log(std_of_classes).sum(axis=1) - norm / 2
