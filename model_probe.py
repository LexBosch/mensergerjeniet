"""
This code can be used to learn about the strategies employed by different models.

A scenario is defined as a list of 17 integers where:
    - The first 4 indicate the positions of the pawns of the acting player
    - The next 12 indicate the positions of the pawns of the other players, in order
    - 0 is the undeployed position
    - -1 to -4 are the home base (last positions)
    - All positions are relative to the first player
    - The last integer indicates the dice throw (1-6)

Author: Cas
"""
import os
import pickle
from collections import Counter
from typing import List, Dict

import numpy as np

from model import Model

"""
Scenario's to inspect behaviour when a pawn is yet to be deployed.
"""
START_POSITIONS = [
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 1],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 2],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 3],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 4],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 5],
    [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 6],
]

"""
Scenario's where the player is able to hit another pawn.
"""
HIT_PAWN = [
    [12,0,0,0,    15,0,0,0,   24,0,0,0,   33,0,0,0, 3],    # 1
    [4,10,0,0,    12,15,0,0,  27,22,0,0,  31,35,0,0, 2],   # 2
    [5,13,31,-4,  1,24,-3,0,  8,2,25,34,  22,20,-2,0, 3],  # 1 or 3
    [-2,-1,13,19, -4,-1,-2,0, 0,17,-3,28, -1,34,22,0, 4],  # 3
    [0,3,0,0,     0,9,0,0,    0,25,0,0,   0,8,0,0, 5],     # 2
    [0,15,0,30,   -1,0,0,12,  36,0,12,0,  -2,0,-3,0, 6],   # 4
]

"""
Scenario's where the player can choose between moving an existing pawn or deploying a new one.
"""
NEW_PAWN = [
    [12,0,0,0,   0,0,0,0,    0,0,0,0,    0,0,0,0, 6],
    [24,0,0,0,   0,12,0,0,   0,0,26,0,   36,0,0,0, 6],
    [0,6,0,0,    0,14,0,0,   0,0,22,0,   0,0,0,32, 6],
    [0,8,0,0,    0,0,15,0,   0,0,24,0,   35,0,0,0, 6],
    [0,12,14,0,  23,0,0,26,  32,35,0,0,  0,0,4,12, 6],
    [0,23,17,36, -2,-4,25,0, 12,15,0,-1, -3,0,-2,0, 6],
]

"""
Scenario's where the player can enter their home base (end positions).
"""
ENTER_BASE = [
    [39,0,0,0,    0,0,0,0,   0,0,0,0,    0,0,0,0, 1],
    [0,39,0,0,    0,0,13,0,  22,0,0,0,   36,0,0,0, 2],
    [12,0,38,0,   23,0,0,12, 34,0,23,0,  11,0,-3,0, 3],
    [38,-2,0,0,   24,-4,0,0, 0,0,-4,12,  0,-3,0,0, 4],
    [5,0,13,37,   0,-2,0,-4, 12,0,33,-1, 34,0,24,0, 5],
    [36,13,24,30, 1,35,-4,0, 13,0,26,-4, 0,-2,14,25, 6],
]

"""
Scenario's where the player can choose to put their pawn in an exposed position.
"""
HITTABLE_POSITION = [
    [23,0,5,0,   0,6,0,0,   0,0,0,0,  0,0,0,0, 2],   # 1
    [1,0,5,0,    0,6,0,0,   0,0,0,0,  0,0,0,0, 2],   # 1
    [16,0,25,0,  0,0,26,0,  0,33,0,0, 0,0,35,0, 3],  # 1
    [16,-4,-2,5, 0,18,0,35, 0,0,33,0, 0,0,22,0, 4],  # 4
    [0,32,12,0,  0,22,0,-3, 0,0,34,0, 0,0,25,0, 5],  # 3
    [0,27,5,0,   0,0,0,-3,  -2,0,0,0, 0,31,0,0, 6],  # 1 or 3 or 4
]


def load_generation(filepath: str) -> List[Model]:
    """
    Loads in a generation of models of arbitrary length from their pickled weights and biases.

    :param filepath: Pickle file of the weights and biases
    :return: List of initialized models
    """
    with open(filepath, 'rb') as input_handle:
        wandb = pickle.load(input_handle)

    models = []

    for i in range(len(wandb)):
        models.append(Model(layers=wandb[i]['layers'], biases=wandb[i]['biases']))

    return models


def load_random_models(length=100) -> List[Model]:
    """
    Returns a list of randomly initialized models.

    :param length: Specifies the amount of models to return
    :return: List of initialized models
    """
    random_models = []

    for i in range(length):
        random_models.append(Model())

    return random_models


def test_scenario(models: List[Model], scenario: List[int]) -> Counter:
    """
    Runs a test scenario on all given models and summarises the results.

    :param models: List of models
    :param scenario: Valid scenario as defined above
    :return: Counter object of the chosen pawns
    """
    move_choices = []

    for i in range(len(models)):
        move_choices.append(np.argmax(models[i].pick_move(np.array(scenario))))

    move_choices = Counter(move_choices)

    return move_choices


def collect_filenames(directory):
    """
    Collects all the pickle filenames in the given directory.

    :param directory: Directory to collect the files from
    :return: List of absolute paths to the files contained in the directory
    """
    return [os.path.abspath(f'{directory}\\{path}') for path in os.listdir(directory)]


def main(test_type: List[List[int]],
         weights_dir: str = 'weights',
         random_baseline_amount: int = 100) -> Dict[str, List[Counter]]:
    """
    Main testing pipeline that collects results of all scenario's of a given category over all model generations
    contained in the specified directory.

    A generation of randomly initialized models is included too so any possible bias in the model architecture can be
    studied.

    :param test_type: The collection of scenario's to test against
    :param weights_dir: Directory containing the pickled weights and biases
    :param random_baseline_amount: Number of random models to initialize
    :return: Dictionary of Listed Counter objects in order of the scenario's
    """
    filenames = collect_filenames(weights_dir)

    models = {}

    for filename in filenames:
        models[filename.split(os.sep)[-1].replace('.pickle', '')] = load_generation(filename)

    models['random'] = load_random_models(length=random_baseline_amount)

    results = {}

    for name in models.keys():
        result = []
        for scenario in test_type:
            result.append(test_scenario(models[name], scenario))

        results[name] = result

    return results


if __name__ == '__main__':
    results = main(HIT_PAWN)

    for key in results.keys():
        print(f'Results for generation {key}')
        print(results[key])
