import numpy as np


def feature_sum(input_state, indices, dependency_indices):
    a, b, c = dependency_indices
    input_state[indices] = input_state[a] + input_state[b] - input_state[c]
    return input_state


def normalize(input_state, indices, dependency_indices):
    total_sum = np.sum([input_state[i] for i in dependency_indices])

    if total_sum != 0:
        for i in indices:
            input_state[i] = input_state[i] / total_sum

    return input_state
