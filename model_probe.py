import pickle
from model import Model
import numpy as np
from collections import Counter


def load_generation(filepath):
    with open(filepath, 'rb') as input_handle:
        wandb = pickle.load(input_handle)

    models = []
    random_models = []

    for i in range(len(wandb)):
        models.append(Model(layers=wandb[i]['layers'], biases=wandb[i]['biases']))
        random_models.append(Model())

    move_choices = []
    random_move_choices = []
    scenario = [5,0,0,0, 1,0,0,0, 4,0,0,0, 0,0,0,0, 4]

    for i in range(len(models)):
        move_choices.append(np.argmax(models[i].pick_move(np.array(scenario))))
        random_move_choices.append(np.argmax(random_models[i].pick_move(np.array(scenario))))

    move_choices = Counter(move_choices)
    random_move_choices = Counter(random_move_choices)

    pass


if __name__ == '__main__':
    load_generation('weights_100_500_300.pickle')
