from .NeuralNetwork.NeuralNetwork import NeuralNetwork


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.learn([[0]], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

