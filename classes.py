from math import exp
from random import uniform, sample, shuffle
import numpy as np
from numpy import array, add, e, matmul, argmax, empty, copy, zeros
from pickle import dump, HIGHEST_PROTOCOL, load
import scipy.io
import scipy.stats
from scipy.special import softmax

train_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['train']


# utils
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))


# Wrapper for raw data
class DataSet(list):
    def __init__(self, raw):
        self.data = raw[0, 0]
        self.idx = 0
        self.input_len = len(self.data['images'][0, 0][0])

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data['images'][0, 0])

    def __next__(self):
        self.idx += 1
        try:
            # return scipy.stats.zscore(self.data['images'][0, 0][self.idx - 1]), self.data['labels'][0, 0][self.idx - 1]
            return self.data['images'][0, 0][self.idx - 1] / 255, self.data['labels'][0, 0][self.idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration

    def shuffle(self):
        shuffle(self.data)

# Neural Network
class Network:
    def __init__(self, hidden_layers, labels, raw):
        self.layers = []
        # Convert raw data
        self.data_set = DataSet(raw)
        # print(self.data_set.data)
        # Input layer
        self.layers.append(Base_Layer(self.data_set.input_len))
        # Hidden layers
        for i in range(len(hidden_layers)):
            self.layers.append(Hidden_Layer(hidden_layers[i], self.layers[-1]))
        # Output Layer
        self.layers.append(Output(len(labels), self.layers[-1], labels))
        self.weights = [[]] + [layer.weights for layer in self.layers[1:]]
        self.biases = [[]] + [layer.biases for layer in self.layers[1:]]
        self.zs = [[]] + [layer.zs for layer in self.layers[1:]]
        self.nodes = [layer.nodes for layer in self.layers]
        self.w_batch = []
        self.b_batch = []
        self.counter = 1
        self.learning_rate = 0.1
        self.batch_size = 20
        self.learning_rate = self.learning_rate / self.batch_size
        self.batch_cost = 0
        self.batch = []

    def run(self, data):
        self.layers[0].nodes = array([data]).transpose()
        for layer in self.layers:
            layer.eval()
        # print(self.layers[-1].nodes)

    def train(self, epochs):
        dataset_size = 60000
        for i in range(epochs):
            self.counter = 0
            self.data_set.idx = 0
            print('Welcome to a new epoch!')
            self.data_set.shuffle()
            while True:
                data, actual = next(self.data_set)
                self.run(data)
                # if self.counter == 1:
                #     for layer in self.layers: print(layer.nodes)
                self.calc_change(actual)
                self.counter += 1
                # Back prop and reset batch
                if not self.counter % self.batch_size:
                    self.back_prop()
                    print(f'Epoch: {i}')
                    print(actual, self.layers[-1].result, self.layers[-1].nodes)
                if self.counter >= dataset_size:
                    break
        with open('zerolayers.pickle', 'wb') as file:
            dump(self, file, protocol=HIGHEST_PROTOCOL)


    def test(self, raw):
        dataset = DataSet(raw)
        correct = 0
        # print(dataset.data)
        # print(next(dataset))
        while True:
            try:
                data, answer = next(dataset)
                self.run(data)
                print(self.layers[-1].result, answer[0])
                if answer[0] == self.layers[-1].result:
                    correct += 1
            except:
                break
        print(correct)
        return correct, len(dataset)

    def calc_change(self, actual):
        n = len(self.layers)
        desired = array([[1 if i == actual else 0 for i in range(10)]]).transpose()
        weight_changes = [zeros(array(w).shape) for w in self.weights]
        bias_changes = [zeros(array(b).shape) for b in self.biases]
        dCdz = np.multiply(dsigmoid(self.zs[-1]), self.layers[-1].nodes - desired)
        bias_changes[-1] = dCdz
        weight_changes[-1] = dCdz @ self.layers[-2].nodes.transpose()
        # print(dCdz)
        # print(self.layers[-2].nodes.transpose())
        # print(weight_changes[-1])
        for i in range(n - 2, 0, -1):
            z = self.zs[i]
            dCdz = np.multiply(dsigmoid(z), self.weights[i + 1].transpose() @ dCdz)
            weight_changes[i] = dCdz @ self.nodes[i - 1].transpose()
            bias_changes[i] = dCdz
        self.w_batch.append(weight_changes)
        self.b_batch.append(bias_changes)
        for i in range(10):
            self.batch_cost += (self.layers[-1].nodes[i] - desired[i]) ** 2


    def back_prop(self):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for j in range(len(self.w_batch)):
                layer.weights = add(layer.weights, -self.learning_rate * array(self.w_batch[j][i]))
                layer.biases = add(layer.biases, -self.learning_rate * array(self.b_batch[j][i]))
        self.w_batch = []
        self.b_batch = []
        print(f"Batch {self.counter // self.batch_size} average cost: {self.batch_cost / self.batch_size}")
        self.batch_cost = 0

    def randomize_batch(self):
        self.batch = sample(range(0, 59999))


class Base_Layer:
    def __init__(self, _num_nodes):
        self.num_nodes = _num_nodes
        self.nodes = array([[0 for i in range(_num_nodes)]]).transpose()
        self.zs = array([[0 for i in range(_num_nodes)]]).transpose()
    def eval(self):
        pass


class Hidden_Layer(Base_Layer):
    def __init__(self, _num_nodes, _prev_layer):
        super().__init__(_num_nodes)
        self.prev_layer = _prev_layer
        self.weights = array([[np.random.normal(scale = self.prev_layer.num_nodes ** (-0.5)) for weight in range(_prev_layer.num_nodes)] for node in range(_num_nodes)])
        self.biases = array([[0 for node in range(_num_nodes)]]).transpose()

    def eval(self):
        self.zs = add(self.weights @ self.prev_layer.nodes, self.biases)
        self.nodes = sigmoid(self.zs)


class Output(Hidden_Layer):
    def __init__(self, _num_nodes, _prev_layer, labels):
        super().__init__(_num_nodes, _prev_layer)
        self.labels = labels
        self.result = None

    def eval(self):
        super().eval()
        self.result = self.labels[argmax(self.nodes)]


# net = Network([5],[0,1,2,3,4,5,6,7,8,9],train_set)
# net.train(3)
with open("zerolayers.pickle", "rb") as file:
    net = load(file)
    net.train(20)
