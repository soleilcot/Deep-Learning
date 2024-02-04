from base import BaseModel
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models


class BasicNeuralNetwork(BaseModel):
    def build_model(self):
        self.model = models.Sequential(
            [
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(10, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, epochs=5):
        self.model.fit(self.train_x, self.train_y, epochs)

    def evaluate(self):
        self.model.evaluate(self.test_x, self.train_y)
