from typing import Optional, cast, Literal, TypeVar

import numpy as np
from numpy import ndarray, dtype, int64, uint8, float64

import linearregression as lr
import template
from resourceloader import Resources, Attack

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)


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


def profile(res: Resources, pois: ndarray[tuple[B], dtype[int64]])\
        -> Optional[tuple[template.Model[B], lr.Model[B]]]:

    profile_data = res.load_profile('profile')
    if profile_data is None:
        return print('no profile file found')
    if profile_data['labels'].shape[0] != pois.size:
        return print('profile and pois have different block counts')
    labels = cast(ndarray[tuple[B, int], dtype[uint8]], profile_data['labels'])
    traces = cast(ndarray[tuple[B, int], dtype[float64]], profile_data['traces'][np.arange(pois.size), :, pois])
    del profile_data

    return template.Model[B](labels, traces), lr.Model[B](labels, traces)


def attack(res: Resources, pois: ndarray[tuple[B], dtype[int64]]) -> None:
    temp = profile(res, pois)
    if temp is None:
        return
    template_model, lr_model = temp
    del temp

    plaintexts: ndarray[tuple[B, int], dtype[uint8]]
    traces: ndarray[tuple[B, int], dtype[float64]]

    attack_data = res.load_profile('attack')
    if attack_data is None:
        return print('no attack file found')
    if attack_data['traces'].shape[0] != pois.size:
        return print('attack and pois have different block counts')
    plaintexts = cast(ndarray[tuple[B, int], dtype[uint8]], attack_data['labels'])
    traces = cast(ndarray[tuple[B, int], dtype[float64]], attack_data['traces'][np.arange(pois.size), :, pois])
    del attack_data

    np.set_printoptions(formatter={'int': hex})
    print('template keys:')
    print(template_model.get_key(plaintexts, traces), '\n')
    print('lr keys:')
    print(lr_model.get_key(plaintexts, traces), '\n')


def main() -> None:
    res = Resources.new('resources.json')
    if res is None:
        return print('no resources.json file found')

    snr_pois: ndarray[tuple[Literal[16]], dtype[int64]]
    poi = res.load('poi')
    if poi is None:
        return print('no poi file found')
    if not is_ndarray(poi['snr_pois'].squeeze(), (16, ), int64):
        return print('snr_pois is ill-formed')
    snr_pois = poi['snr_pois'].squeeze()
    del poi
    attack(res, snr_pois)


if __name__ == '__main__':
    main()
