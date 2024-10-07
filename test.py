import numpy as np


def test():
    print(np.broadcast_to(
        np.repeat(np.arange(start=0, stop=256, dtype=np.uint8), 4),
        (16, 256 * 4)
    ).dtype)


if __name__ == '__main__':
    test()
