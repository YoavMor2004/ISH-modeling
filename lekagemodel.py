from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, cast

import numpy as np
from numpy import ndarray, dtype, uint8, float64

N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)
K = TypeVar('K', bound=int)


class LeakageModel(ABC, Generic[P]):
    @abstractmethod
    def keys_probability(
            self,
            plaintexts: ndarray[tuple[N], dtype[uint8]],
            traces: ndarray[tuple[N, P], dtype[float64]],
            keys: ndarray[tuple[K], dtype[uint8]]
    ) -> ndarray[tuple[K, P], dtype[float64]]:
        pass

    def get_key(
            self,
            plaintexts: ndarray[tuple[Literal[16], N],    dtype[uint8]],
            traces:     ndarray[tuple[Literal[16], N, P], dtype[float64]]
    ) -> ndarray[tuple[P], dtype[uint8]]:

        return self.keys_probability(
            cast(ndarray[tuple[N], dtype[uint8]], plaintexts[0, :]),
            cast(ndarray[tuple[N, P], dtype[float64]], traces[0, :, :]),
            np.arange(256, dtype=uint8)
        ).argmax(axis=0)
