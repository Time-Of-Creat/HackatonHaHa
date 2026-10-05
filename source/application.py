from source.data_management.hockey_data_generator import HockeyDataGenerator
from source.neural_network.neural_network import NeuralNetwork


class Application:
    def __init__(self):
        self.data_generator = HockeyDataGenerator()
        self.nn = NeuralNetwork()

    def learn_and_validate(self):
        train_data = self.data_generator.get_train_data()
        validation_data = self.data_generator.get_val_data()

        self.nn.fit(train_data[0], train_data[1])
        self.nn.validate(validation_data[0], validation_data[1])

    def form_predictions_table(self):
        for game in range(86, 103):
            for zone in range(1, 9):
                game_data = self.data_generator.get_match_zone_features(f"M{game:03d}", str(zone))
                prediction = int(
                    self.nn.predict(game_data)[0] * self.data_generator.encoder.get_capacity(f"M{game:03d}", str(zone)))
