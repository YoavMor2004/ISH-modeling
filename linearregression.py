from typing import TypeVar, Literal, reveal_type, cast

import numpy as np
from numpy import ndarray, dtype, float64, uint8

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


def model(labels: ndarray[tuple[N], dtype[uint8]], traces: ndarray[tuple[N, P], dtype[float64]])\
        -> ndarray[tuple[Literal[9], P], dtype[float64]]:

    x: ndarray[tuple[Literal[9], N], dtype[uint8]]
    x = expand(cast(
        ndarray[tuple[N, Literal[1]], dtype[uint8]],
        labels[:, None]
    )).squeeze(axis=2)
    return np.linalg.inv(np.matmul(x, x.T, dtype=np.int64)) @ x @ traces


def loss(
        traces:       ndarray[tuple[N, P],          dtype[float64]],
        c:            ndarray[tuple[N, K],          dtype[uint8]],
        coefficients: ndarray[tuple[Literal[9], P], dtype[float64]]
) -> ndarray[tuple[K, P], dtype[float64]]:

    return np.square(traces[None, :, :] - expand(c).transpose(2, 1, 0) @ coefficients).sum(axis=1)


def keys_probability(
        plaintexts: ndarray[tuple[N], dtype[uint8]],
        traces:     ndarray[tuple[N, P], dtype[float64]],
        keys:       ndarray[tuple[K], dtype[uint8]],
        m:          ndarray[tuple[Literal[9], P], dtype[float64]]
) -> ndarray[tuple[K, P], dtype[float64]]:

    return loss(traces, keys[None, :] ^ plaintexts[:, None], m)
