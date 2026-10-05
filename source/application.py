from source.data_management.hockey_data_generator import HockeyDataGenerator
from source.neural_network.neural_network import NeuralNetwork


class Application:
    def __init__(self):
        self.data_generator = HockeyDataGenerator()
        self.nn = NeuralNetwork()

    def get_game_zone_prediction(self, game: str, zone: str):
        game_data = self.data_generator.get_match_zone_features(game, zone)
        prediction = int(
            self.nn.predict(game_data)[0] * self.data_generator.encoder.get_capacity(game, zone))
        return prediction
