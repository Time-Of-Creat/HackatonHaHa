from source.neural_network.neural_network import NeuralNetwork
from source.data_management.hockey_data_generator import HockeyDataGenerator


if __name__ == "__main__":
    data_generator = HockeyDataGenerator("../data")
    nn = NeuralNetwork()

    train_data = data_generator.get_train_data()
    validation_data = data_generator.get_val_data()

    nn.fit(train_data[0], train_data[1])
    nn.validate(train_data[0], train_data[1])

