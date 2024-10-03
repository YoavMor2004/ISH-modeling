import numpy as np
from numpy import uint8

import linearregression as lr


def test():
    a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    print(a)
    print('\n')
    print(np.linalg.inv(a))


if __name__ == '__main__':
    test()
