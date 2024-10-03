import numpy as np
from numpy import uint8

import linearregression as lr


def test():
    lr.model(np.array([3, 5], dtype=uint8), None)


if __name__ == '__main__':
    test()
