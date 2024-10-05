import numpy as np


def test():
    x = np.broadcast_to(np.arange(256, dtype=np.uint8), (4, 256))


if __name__ == '__main__':
    test()
