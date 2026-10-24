import numpy as np


def test_data_handling():
    birds = DetectedBirds.get()

    store_birds(birds)

    entires = get_data()

    for entry in entires:
        assert entry['species'] == 'sparrow'
        assert entry['confidence'] == 0.95
        assert entry['timestamp'] is not None