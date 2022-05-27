import numpy as np

class Model():
    def __init__(self, dimensions=[17,10,5], output='softmax', layers=None, biases=None):
        self.layers = []
        self.biases = []
        self.fitness_history = []
        self.iteration_history = []
        self.output = self._activation(output)
        self.fitness = 0
        self.current_iteration = 0
        self.dimensions = dimensions

        # Initialize neuron weights if they were not supplied
        if layers == None:
            for i in range(len(dimensions)-1):
                shape = (dimensions[i], dimensions[i+1])
                std = np.sqrt(2 / sum(shape))
                layer = np.random.normal(0, std, shape)
                self.layers.append(layer)
        else:
            self.layers = layers

        # Initialize neuron biases if they were not supplied
        if biases == None:
            for i in range(len(dimensions)-1):
                shape = (dimensions[i], dimensions[i+1])
                std = np.sqrt(2 / sum(shape))
                bias = np.random.normal(0, std, (1,  dimensions[i+1]))
                self.biases.append(bias)
        else:
            self.biases = biases

    def _activation(self, output):
        if output == 'softmax':
            return lambda X : np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X : (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X : X

    def _weight_init(self, method):
        pass

    def update_fitness(self, iteration, value):
        if iteration != self.current_iteration:
            self.fitness_history.append(self.fitness)
            self.iteration_history.append(iteration)
            self.fitness = 0
            self.current_iteration = iteration
        self.fitness += value

    def pick_move(self, X):
        X = np.array([X])
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X) # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU
        return X[0]

if __name__ == '__main__':
    dims = [17, 4]
    m = Model(dimensions=dims)
    print(m.pick_move(np.array([[8,0,0,0,7,0,0,0,6,0,0,0,5,0,0,0,3]])))