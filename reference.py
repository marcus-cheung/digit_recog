class network:
    def __init__(self):
        self.weights_matrices = [None] * 3
        self.bias_matrices = [None] * 3
        self.initialize_weights()
        self.randomize_weights()

    def initialize_weights(self):
        self.weights_matrices[0] = np.zeros((16, 784))
        self.weights_matrices[1] = np.zeros((16, 16))
        self.weights_matrices[2] = np.zeros((10, 16))

        self.bias_matrices[0] = np.zeros(16)
        self.bias_matrices[1] = np.zeros(16)
        self.bias_matrices[2] = np.zeros(10)

    def randomize_weights(self):
        for weights in self.weights_matrices:
            for row in weights:
                for i in range(len(row)):
                    row[i] = randint(1, 100) / 1000
        for biases in self.bias_matrices:
            for i in range(len(biases)):
                biases[i] = randint(1, 100) / 1000

    def evaluate(self, image) -> list[int]:
        layers = [None] * 4
        layers[0] = image
        for i in range(3):
            # print('Evaluating layer: ' + str(i))
            # print(self.weights_matrices[i])
            # print(layers[i])
            layers[i + 1] = msigmoid(np.add(np.matmul(self.weights_matrices[i], layers[i]), self.bias_matrices[i]))
        return layers[-1]

    def evaluate_nodes(self, image):
        nodes = [None] * 4
        nodes[0] = image
        for i in range(3):
            nodes[i + 1] = msigmoid(np.add(np.matmul(self.weights_matrices[i], nodes[i]), self.bias_matrices[i]))
        return nodes

    def cost(self, image) -> int:
        cost = 0
        output = self.evaluate(image)
        correct = 100  # TODO: Match with label
        for i in range(10):
            if i == correct:
                cost += (1 - output[i]) ** 2
            else:
                cost += output[i] ** 2
        return cost

    def batch_cost(self, batch) -> int:
        total_cost = 0
        for image in batch:
            total_cost += self.cost(image)
        return total_cost

    def img_backpropagate(self, image):
        nodes = self.evaluate_nodes(image)
        desired = [None] * 4
        # desired[-1] = nodes[-1] this is wrong should be the correct answer
        wadjs = self.weights[0:]
        badjs = self.biases[0:]
        for i in range(3):
            L = 3 - i
            for j in range(len(nodes[L])):
                # Calculate dC/dw
                for k in range(len(nodes[L - 1])):
                    wadjs[L][j][k] = nodes[L - 1][k] * nodes[L][j] * (1 - nodes[L][j]) * 2 * (nodes[L][j] - desired[L][j])
                # Calculate dC/db
                    badjs[L][j] =  nodes[L][j] * (1 - nodes[L][j]) * 2 * (nodes[L][j] - desired[L][j])
            if i < 2:
                # Calculating dC/da_k
                for k in range(len(nodes[L - 1])):
                    desired[L - 1][k] = 0
                    for j in range(len(nodes[L])):
                        desired[L - 1][k] += self.weights_matrices[L - 1][j][k] * nodes[L][j] * (1 - nodes[L][j]) * 2 * (nodes[L][j] - desired[L][j])
        return wadjs, badjs

    def batch_backpropagate(self, batch):
        N = len(batch)
        wadjs = self.weights[0:]
        badjs = self.biases[0:]
        for image in batch:
            wadjs_add, badjs_add = self.img_backpropagate(image)
            for i in range(3):
                wadjs[i] = np.add(wadjs[i], wadjs_add[i])
                badjs[i] = np.add(badjs[i], badjs_add[i])
        for i in range(3):
            self.weights_matrices[i] = np.add(self.weights_matrices[i], wadjs[i] / N)
            self.bias_matrices[i] = np.add(self.bias_matrices[i], badjs[i] / N)