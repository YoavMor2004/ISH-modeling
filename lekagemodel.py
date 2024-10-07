from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, cast

import numpy as np
from numpy import ndarray, dtype, uint8, float64

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
K = TypeVar('K', bound=int)


class LeakageModel(ABC, Generic[P]):
    @abstractmethod
    def keys_probability(
            self,
            plaintexts: ndarray[tuple[B, N],    dtype[uint8]],
            traces:     ndarray[tuple[B, N, P], dtype[float64]],
            keys:       ndarray[tuple[B, K],       dtype[uint8]]
    ) -> ndarray[tuple[B, K, P], dtype[float64]]:
        pass

    def get_key(
            self,
            plaintexts: ndarray[tuple[B, N],    dtype[uint8]],
            traces:     ndarray[tuple[B, N, P], dtype[float64]]
    ) -> ndarray[tuple[B, P], dtype[uint8]]:

        return self.keys_probability(
            plaintexts,
            traces,
            cast(
                ndarray[tuple[B, Literal[256]], dtype[uint8]],
                np.broadcast_to(np.arange(256, dtype=uint8), (plaintexts.shape[0], 256))
            )
        ).argmax(axis=1)
