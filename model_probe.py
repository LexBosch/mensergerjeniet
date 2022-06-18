import pickle
from model import Model
import numpy as np
from collections import Counter
import os


START_POSITIONS = [
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 2],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 3],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 4],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 5],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 6],
]

HIT_PAWN = [
    [12,0,0,0,    15,0,0,0,   24,0,0,0,   33,0,0,0, 3],  # 1
    [4,10,0,0,    12,15,0,0,  27,22,0,0,  31,35,0,0, 2],  # 2
    [5,13,31,-4,  1,24,-3,0,  8,2,25,34,  22,20,-2,0, 3],  # 1 or 3
    [-2,-1,13,19, -4,-1,-2,0, 0,17,-3,28, -1,34,22,0, 4],  # 3
    [0,3,0,0,     0,9,0,0,    0,25,0,0,   0,8,0,0, 5],  # 2
    [0,15,0,30,   -1,0,0,12,  36,0,12,0,  -2,0,-3,0, 6],  # 4
]

NEW_PAWN = [
    [12,0,0,0,   0,0,0,0,    0,0,0,0,    0,0,0,0, 6],
    [24,0,0,0,   0,12,0,0,   0,0,26,0,   36,0,0,0, 6],
    [0,6,0,0,    0,14,0,0,   0,0,22,0,   0,0,0,32, 6],
    [0,8,0,0,    0,0,15,0,   0,0,24,0,   35,0,0,0, 6],
    [0,12,14,0,  23,0,0,26,  32,35,0,0,  0,0,4,12, 6],
    [0,23,17,36, -2,-4,25,0, 12,15,0,-1, -3,0,-2,0, 6],
]

ENTER_BASE = [
    [39,0,0,0,    0,0,0,0,   0,0,0,0,    0,0,0,0, 1],
    [0,39,0,0,    0,0,13,0,  22,0,0,0,   36,0,0,0, 2],
    [12,0,38,0,   23,0,0,12, 34,0,23,0,  11,0,-3,0, 3],
    [38,-2,0,0,   24,-4,0,0, 0,0,-4,12,  0,-3,0,0, 4],
    [5,0,13,37,   0,-2,0,-4, 12,0,33,-1, 34,0,24,0, 5],
    [36,13,24,30, 1,35,-4,0, 13,0,26,-4, 0,-2,14,25, 6],
]

HITTABLE_POSITION = [
    [23,0,5,0,   0,6,0,0,   0,0,0,0,  0,0,0,0, 2],
    [1,0,5,0,    0,6,0,0,   0,0,0,0,  0,0,0,0, 2],
    [16,0,25,0,  0,0,26,0,  0,33,0,0, 0,0,35,0, 3],
    [16,-4,-2,5, 0,18,0,35, 0,0,33,0, 0,0,22,0, 4],
    [0,32,12,0,  0,22,0,-3, 0,0,34,0, 0,0,25,0, 5],
    [0,27,5,0,   0,0,0,-3,  -2,0,0,0, 0,31,0,0, 6],
]


def load_generation(filepath):
    with open(filepath, 'rb') as input_handle:
        wandb = pickle.load(input_handle)

    models = []

    for i in range(len(wandb)):
        models.append(Model(layers=wandb[i]['layers'], biases=wandb[i]['biases']))

    return models


def load_random_models():
    random_models = []

    for i in range(100):
        random_models.append(Model())

    return random_models


def test_scenario(models, scenario):
    move_choices = []

    for i in range(len(models)):
        move_choices.append(np.argmax(models[i].pick_move(np.array(scenario))))

    move_choices = Counter(move_choices)

    return move_choices


def collect_filenames(directory):
    return [os.path.abspath(f'{directory}\\{path}') for path in os.listdir(directory)]


def main(test_type):
    filenames = collect_filenames('weights')

    models = {}

    for filename in filenames:
        models[filename.split(os.sep)[-1].replace('.pickle', '')] = load_generation(filename)

    models['random'] = load_random_models()

    results = {}

    for name in models.keys():
        result = []
        for scenario in test_type:
            result.append(test_scenario(models[name], scenario))

        results[name] = result

    pass



if __name__ == '__main__':
    main(HIT_PAWN)
