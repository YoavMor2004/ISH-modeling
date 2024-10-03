from functools import reduce

import numpy as np

import resources
import linearregression as lr
import template


def main():
    res = resources.new('resources.json')
    if res is None:
        return print('no resources.json file found')

    profile = resources.load(res, 'profile')
    poi = resources.load(res, 'poi')

    del res

    keys = profile['keys'].squeeze()
    plaintexts = profile['plaintexts'].squeeze()
    traces = profile['traces'].squeeze()[:, 673][:, np.newaxis]

    del profile

    pois = reduce(np.union1d, (poi['snr_pois'], poi['lda_pois'], poi['pca_pois']))

    del poi

    labels = keys ^ plaintexts

    del keys
    del plaintexts

    # print(lr.model(labels, traces))
    if labels.ndim != 1 or traces.ndim != 2:
        return print('data is of the wrong dimensions')
    if labels.shape[0] != traces.shape[0]:
        return print('dimensions of data do not fit')
    if labels.dtype != np.uint8 or traces.dtype != np.float64:
        return print('data is of the wrong data type')
    print(traces)
    print(template.model(labels, traces).std_of_classes)


if __name__ == '__main__':
    main()

