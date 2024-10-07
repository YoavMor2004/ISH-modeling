from typing import Optional, Literal, TypeVar, Iterator, NamedTuple, Self, Generic, cast, Iterable

import numpy as np
from numpy import ndarray, dtype, int64, uint8, float64

import linearregression as lr
import template
from lekagemodel import LeakageModel
from resourceloader import Resources

B = TypeVar('B', bound=int)
N = TypeVar('N', bound=int)
P = TypeVar('P', bound=int)


def tuple_str(t: tuple) -> str:
    return '[' + ']['.join(map(str, t)) + ']'


def ndarray_repr(x: ndarray[tuple, dtype], /, name: Optional[str] = None) -> str:
    if name is None:
        return f'{x.dtype}{tuple_str(x.shape)}'
    return f'{x.dtype} {name}{tuple_str(x.shape)}'


def is_compatible_shape(x: ndarray[tuple, dtype], shape: tuple[Optional[int], ...], /) -> bool:
    # _TODO: Double pointer non-deterministic algorithm for weeding out 1-dimensions
    return all((
        len(shape) <= x.ndim,
        all(d == 1 for d in x.shape[len(shape):]),
        all(shape[d] in {x.shape[d], None} for d in range(len(shape)))
    ))


def is_valid_ndarray(x: ndarray[tuple, dtype], shape: tuple[Optional[int], ...], dt: type, /) -> bool:
    return is_compatible_shape(x, shape) and x.dtype == dt


class Data(NamedTuple, Generic[B, N]):
    profiling_labels:     ndarray[tuple[B, N], dtype[uint8]]
    profiling_traces:     ndarray[tuple[B, N], dtype[float64]]
    attacking_plaintexts: ndarray[tuple[B, N], dtype[uint8]]
    attacking_traces:     ndarray[tuple[B, N], dtype[float64]]

    @classmethod
    def load(cls, resources_filepath, block_count: B, trace_count: N) -> Self | str:
        profiling_labels:     ndarray[tuple[B, N], dtype[uint8]]
        profiling_traces:     ndarray[tuple[B, N], dtype[float64]]
        attacking_plaintexts: ndarray[tuple[B, N], dtype[uint8]]
        attacking_traces:     ndarray[tuple[B, N], dtype[float64]]

        res = Resources.new(resources_filepath)
        if res is None:
            return 'resources file not found'

        poi_file = res.load('poi_file')
        if poi_file is None:
            return 'poi file not found'
        if res['poi'] not in poi_file:
            return 'poi not found'
        if not is_valid_ndarray(poi_file[res['poi']].squeeze(), (block_count,), int64):
            return 'poi ndarray is not properly formed'
        poi: ndarray[tuple[B], dtype[int64]] = poi_file[res['poi']].squeeze()
        del poi_file

        profile_file = res.load('profile')
        if profile_file is None:
            return 'profile file not found'
        if 'labels' not in profile_file:
            return 'profiling labels not found'
        if 'traces' not in profile_file:
            return 'profiling traces not found'
        if not is_valid_ndarray(profile_file['labels'], (block_count, trace_count), uint8):
            return 'profiling labels ndarray is not properly formed'
        if not is_valid_ndarray(profile_file['traces'], (block_count, trace_count, None), float64):
            return 'profiling traces ndarray is not properly formed'
        profiling_labels = profile_file['labels'].reshape(block_count, trace_count)
        profiling_traces = profile_file['traces'].reshape(block_count, trace_count, -1)[np.arange(block_count), :, poi]
        del profile_file

        attacking_file = res.load('attack')
        if attacking_file is None:
            return 'attacking file not found'
        if 'labels' not in attacking_file:
            return 'attacking plaintexts not found'
        if 'traces' not in attacking_file:
            return 'attacking traces not found'
        if not is_valid_ndarray(attacking_file['labels'], (block_count, trace_count), uint8):
            return 'attacking labels ndarray is not properly formed'
        if not is_valid_ndarray(attacking_file['traces'], (block_count, trace_count, None), float64):
            return 'attacking traces ndarray is not properly formed'
        attack_plaintexts = attacking_file['labels'].reshape(block_count, trace_count)
        attack_traces = attacking_file['traces'].reshape(block_count, trace_count, -1)[np.arange(block_count), :, poi]
        del attacking_file

        return cls(profiling_labels, profiling_traces, attack_plaintexts, attack_traces)


def profile(
        labels: ndarray[tuple[B, N], dtype[uint8]],
        traces: ndarray[tuple[B, N], dtype[float64]]
) -> tuple[template.Model[B], lr.Model[B]]:

    return template.Model[B](labels, traces), lr.Model[B](labels, traces)


def attack(
        plaintexts: ndarray[tuple[B, N], dtype[uint8]],
        traces:     ndarray[tuple[B, N], dtype[float64]],
        models: Iterable[tuple[str, LeakageModel[B]]]
) -> None:

    np.set_printoptions(formatter={'int': hex})
    for name, model in models:
        print(f'{name}:')
        print(model.get_key(plaintexts, traces), '\n')


def main() -> None:
    data = Data[Literal[16], Literal[4096]].load('resources.json', 16, 4096)
    if isinstance(data, str):
        return print(data)
    template_model = template.Model[Literal[16]](data.profiling_labels, data.profiling_traces)
    lr_model = lr.Model[Literal[16]](data.profiling_labels, data.profiling_traces)
    attack(data.attacking_plaintexts, data.attacking_traces, (('template', template_model), ('lr', lr_model)))


if __name__ == '__main__':
    main()
