import numpy as np

def init_param():

    states = ['box1', 'box2', 'box3']

    observations = ['red', 'white']

    start_probability = np.array([0.2, 0.4, 0.4])

    transport_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    seen = ['red', 'white', 'red']

    return states, observations, start_probability, transport_probability, emission_probability, seen