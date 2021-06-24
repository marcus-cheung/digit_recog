from math import exp
from random import uniform
from numpy import array, add, matmul, argmax
import scipy.io
from pickle import dump, HIGHEST_PROTOCOL

train_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['train']

# utils
def sigmoid(x):
    return 1/(1+exp(-x))

def parser(raw):
    data = raw
    return data

class DataSet(list):
    def __init__(self, raw):
        self.data = raw
        self.idx = 0
        self.input_len = len(self.data[0][0])
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.data)
    def __next__(self):
        self.idx += 1
        try:
            return self.data[self.idx-1][0], self.data[self.idx-1][1]
        except IndexError:
            self.idx = 0
            raise StopIteration


class Network:
    def __init__(self, hidden_layers, labels, raw):
        self.layers = []
        # Convert raw data
        self.data_set = DataSet(raw)
        # Input layer
        self.layers.append(Layer(self.data_set.input_len))
        # Hidden layers
        for i in range(len(hidden_layers)):
            self.layers.append(Hidden_Layer(hidden_layers[i], self.layers[-1]))
        # Output Layer
        self.layers.append(Output(len(labels), self.layers[-1], labels))
    def train(self):
        while True:
            try:
                data, answer = next(self.data_gen)
                self.layers[0].nodes = data
                for layer in self.layers:
                    layer.eval()
                self.backpropagate(answer)
            except:
                with open('trained.pickle', 'wb') as file:
                    dump(self, file, protocol=HIGHEST_PROTOCOL)
                break
    def run(self, raw):
        dataset = DataSet(raw)
        correct = 0
        while True:
            try:
                data, answer = next(dataset)
                self.layers[0].nodes = data
                for layer in self.layers:
                    layer.eval()
                if self.layers[-1].result == answer:
                    correct+=1
            except:
                break
        return correct, len(dataset)
        

class Layer:
    def __init__(self, _num_nodes):
        #self.num_nodes = _num_nodes
        self.nodes = array([0 for input in range(_num_nodes)])
    def eval(self):
        pass
    
class Hidden_Layer(Layer):
    def __init__(self, _num_nodes, _prev_layer):
        super().__init__(_num_nodes)
        self.prev_layer = _prev_layer
        self.weights = array([[uniform(0,1) for weight in range(_prev_layer.num_nodes)] for node in range(_num_nodes)])
        self.biases = array([uniform() for node in range(_num_nodes)])
    def eval(self):
        self.nodes = array(map(sigmoid,list(add(matmul(self.weights, self.prev_layer.nodes),self.biases))))

class Output(Hidden_Layer):
    def __init__(self, _num_nodes, _prev_layer, labels):
        super().__init__(_num_nodes, _prev_layer)
        self.labels = labels
        self.result = None
    def eval(self):
        super().eval()
        self.result = self.labels[argmax(self.nodes.flatten)]