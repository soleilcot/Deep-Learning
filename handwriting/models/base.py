class BaseModel:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = None

    def build_model(self):
        raise NotImplementedError("Must implement build_model() in subclass.")

    def train(self, epochs):
        raise NotImplementedError("Must implement train() in subclass.")

    def evaluate(self):
        raise NotImplementedError("Must implement evaluate() in subclass.")

    def display_sample_predictions(self):
        raise NotImplementedError(
            "Must implement display_sample_predictions() in subclass."
        )
