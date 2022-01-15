import scipy.io
from pickle import load
from classes import *
from numpy import argmax

test_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['test']

def main():
    with open("hundred.pickle", "rb") as file:
        trained_network = load(file)
        # for i in range(100):
        #     # data, answer = next(test_set)
        #     # trained_network.run(data)
        #     # print(trained_network.layers[-1].nodes)
        #     # print(trained_network.layers[-1].result)
        #     print(trained_network.layers[-1].result, answer[0])
        correct, total = trained_network.test(test_set)
        print(f'{correct} correct out of {total} cases')

main()