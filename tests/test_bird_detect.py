import numpy as np


def test_bird_detect():
    image = np.zeros((10, 10), dtype=np.uint8)

    result = bird_detect(image)

    assert len(result) == 3