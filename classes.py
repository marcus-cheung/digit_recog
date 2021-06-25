from math import exp
from random import uniform
from numpy import array, add, e, matmul, argmax, empty, copy
from pickle import dump, HIGHEST_PROTOCOL
import scipy.io

train_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['train']


# utils
def sigmoid(x):
    return 1/(1+exp(-x))


# Wrapper for raw data
class DataSet(list):
    def __init__(self, raw):
        self.data = raw[0,0]
        self.idx = 0
        self.input_len = len(self.data['images'][0,0][0])
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.data['images'][0,0])
    def __next__(self):
        self.idx += 1
        try:
            return self.data['images'][0,0][self.idx-1], self.data['labels'][0,0][self.idx-1]
        except IndexError:
            self.idx = 0
            raise StopIteration


# Neural Network
class Network:
    def __init__(self, hidden_layers, labels, raw):
        self.layers = []
        # Convert raw data
        self.data_set = DataSet(raw)
        # Input layer
        self.layers.append(Base_Layer(self.data_set.input_len))
        # Hidden layers
        for i in range(len(hidden_layers)):
            self.layers.append(Hidden_Layer(hidden_layers[i], self.layers[-1]))
        # Output Layer
        self.layers.append(Output(len(labels), self.layers[-1], labels))
        self.w_batch = []
        self.b_batch = []
        self.counter = 1
        # self.learning_rate

    def run(self, data):
        self.layers[0].nodes = data
        for layer in self.layers:
            layer.eval()

    def train(self):
        while True:
            # Back prop and reset batch
            if not self.counter%100:
                self.back_prop()
            try:
                data, actual = next(self.data_set)
                self.run(data)
                self.calc_change(actual)
                print(self.counter)
                self.counter+=1
            except StopIteration:
                try:
                    self.back_prop()
                except:
                    pass
                with open('trained.pickle', 'wb') as file:
                    dump(self, file, protocol=HIGHEST_PROTOCOL)
                break

    def test(self, raw):
        dataset = DataSet(raw)
        correct = 0
        while True:
            try:
                data, answer = next(dataset)
                self.run(data)
                if self.layers[-1].result == answer:
                    correct+=1
            except:
                break
        return correct, len(dataset)

    def calc_change(self, actual):
        n = len(self.layers)
        weight_changes = [None] * n
        bias_changes = [None] * n
        desired = [None] * n
        desired[-1] = [1 if i==actual else 0 for i in range(10)]
        # Layers
        for i in range(n - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            bias_changes[i]  = empty(layer.biases.shape)
            weight_changes[i] = empty(layer.weights.shape)
            desired[i - 1] = copy(prev_layer.nodes)
            # Nodes
            for j in range(len(layer.nodes)):
                node = layer.nodes[j]
                dCdz = node * (1 - node) * 2 * (node - desired[i][j])
                # Calcuating dC/dw
                # Weights
                for k in range(len(prev_layer.nodes)):
                    weight_changes[i][j][k] = prev_layer.nodes[j] * dCdz
                    if i > 0:
                        # Calcuating dC/da
                        desired[i - 1][k] -= layer.weights[j][k] * dCdz 
                # desired[i - 1] = add(desired[i - 1], -dCdz * layer.weights[i][j])
                # Caluating dC/db
                bias_changes[i][j] = dCdz
        if weight_changes is None:
            print('I am the error: ' + str(actual))
        self.w_batch.append(weight_changes)
        self.b_batch.append(bias_changes)

    def back_prop(self):
        training_constant = 0.01
        
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for j in range(len(self.w_batch)):
                layer.weights = add(layer.weights, training_constant * array(self.w_batch[j][i]))
                layer.biases = add(layer.biases, training_constant * array(self.b_batch[j][i]))
        self.w_batch = []
        self.b_batch = []
                
        

class Base_Layer:
    def __init__(self, _num_nodes):
        self.num_nodes = _num_nodes
        self.nodes = array([0 for input in range(_num_nodes)])
    def eval(self):
        pass
    
class Hidden_Layer(Base_Layer):
    def __init__(self, _num_nodes, _prev_layer):
        super().__init__(_num_nodes)
        self.prev_layer = _prev_layer
        self.weights = array([[uniform(0,1) for weight in range(_prev_layer.num_nodes)] for node in range(_num_nodes)])
        self.biases = array([uniform(0,1) for node in range(_num_nodes)])
    def eval(self):
        x = self.weights@self.prev_layer.nodes
        # print(self.prev_layer.nodes)
        self.nodes = array(list(map(sigmoid,list(add(self.weights@self.prev_layer.nodes,self.biases)))))

class Output(Hidden_Layer):
    def __init__(self, _num_nodes, _prev_layer, labels):
        super().__init__(_num_nodes, _prev_layer)
        self.labels = labels
        self.result = None
    def eval(self):
        super().eval()
        self.result = self.labels[argmax(self.nodes)]

net = Network([10,10,10],[0,1,2,3,4,5,6,7,8,9],train_set)
net.train()

