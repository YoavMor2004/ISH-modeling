from functools import reduce
from typing import Optional, cast, Literal

import numpy as np
from numpy import ndarray, dtype, int64, uint8, float64

import linearregression as lr
import template
from resources import Resources


def tuple_str(t: tuple) -> str:
    return '[' + ']['.join(map(str, t)) + ']'


def ndarray_repr(x: ndarray[tuple, dtype], /, name: Optional[str] = None) -> str:
    if name is None:
        return f'{x.dtype}{tuple_str(x.shape)}'
    return f'{x.dtype} {name}{tuple_str(x.shape)}'


def is_ndarray(x: ndarray[tuple, dtype], shape: tuple[Optional[int], ...], dt: type, /) -> bool:
    return (isinstance(x, ndarray) and
            x.ndim == len(shape)
            and all(shape[d] is None or x.shape[d] == shape[d] for d in range(x.ndim))
            and x.dtype == dt)


def profile(res: Resources, pois: ndarray[tuple[int], dtype[int64]])\
        -> Optional[tuple[template.Model[int], ndarray[tuple[Literal[9], int], dtype[float64]]]]:

    labels: ndarray[tuple[int], dtype[uint8]]
    traces: ndarray[tuple[int, int], dtype[float64]]

    profiling_data = res.load('profile')
    if profiling_data is None:
        return print('no profile file found')
    keys = profiling_data['keys'].squeeze()
    plaintexts = profiling_data['plaintexts'].squeeze()
    traces = profiling_data['traces'].squeeze()
    if not is_ndarray(keys, (None,), uint8):
        return print('keys data is ill-formed')
    if not is_ndarray(plaintexts, (keys.size,), uint8):
        return print('plaintexts data is ill-formed')
    if not is_ndarray(traces, (keys.size, None), float64):
        return print('traces data is ill-formed')
    keys = cast(ndarray[tuple[int], dtype[uint8]], keys)
    plaintexts = cast(ndarray[tuple[int], dtype[uint8]], plaintexts)
    labels = keys ^ plaintexts
    traces = cast(ndarray[tuple[int, int], dtype[float64]], traces[:, pois])
    del profiling_data, keys, plaintexts

    return template.model(labels, traces), lr.model(labels, traces)


def f(res: Resources, pois: ndarray[tuple[int], dtype[int64]]) -> None:
    template_model: template.Model
    lr_model: ndarray[tuple[Literal[9], int], dtype[float64]]

    template_model, lr_model = profile(res, pois)

    plaintexts: ndarray[tuple[Literal[16], int], dtype[uint8]]
    traces: ndarray[tuple[Literal[16], int, int], dtype[float64]]

    attack = res.load('attack')
    if attack is None:
        return print('no attack file found')
    plaintexts = attack['plaintexts'].squeeze()
    traces = attack['traces'].squeeze()
    if not is_ndarray(plaintexts, (16, None), uint8):
        return print('plaintexts data is ill-formed')
    if not is_ndarray(traces, (16, plaintexts.shape[1], None), float64):
        return print('traces data is ill-formed')
    plaintexts = cast(ndarray[tuple[Literal[16], int], dtype[uint8]], plaintexts)
    traces = cast(ndarray[tuple[Literal[16], int, int], dtype[float64]], traces[:, :, pois])
    del attack

    np.set_printoptions(formatter={'int': hex})
    print(template.keys_probability(
        cast(ndarray[tuple[int], dtype[uint8]], plaintexts[0, :]),
        cast(ndarray[tuple[int, int], dtype[float64]], traces[0, :, :]),
        np.arange(256, dtype=uint8),
        template_model
    ).argmax(axis=0))

    print(lr.keys_probability(
        cast(ndarray[tuple[int], dtype[uint8]], plaintexts[0, :]),
        cast(ndarray[tuple[int, int], dtype[float64]], traces[0, :, :]),
        np.arange(256, dtype=uint8),
        lr_model
    ).argmax(axis=0))


def main() -> None:
    res = Resources.new('resources.json')
    if res is None:
        return print('no resources.json file found')

    snr_pois: ndarray[tuple[int], dtype[int64]]
    lda_pois: ndarray[tuple[int], dtype[int64]]
    poi = res.load('poi')
    if poi is None:
        return print('no poi file found')
    snr_pois = poi['snr_pois'].squeeze()
    lda_pois = poi['lda_pois'].squeeze()
    if not all(is_ndarray(pois, (None, ), int64) for pois in (snr_pois, lda_pois)):
        return print('pois data is ill-formed')
    del poi
    snr_pois = cast(ndarray[tuple[int], dtype[int64]], snr_pois)
    lda_pois = cast(ndarray[tuple[int], dtype[int64]], lda_pois)

    f(res, lda_pois)
    # f(res, snr_pois)




if __name__ == '__main__':
    main()

