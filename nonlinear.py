class Non_linear():

    def __init__(self, feature):
        self.feature = feature

class Linear(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
    def __str__(self):
        return str(self.feature)
    def evaluate(self, data):
        return data
    def __name__(self):
        return 'Linear'

class Sigmoid(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'sigma(' + str(self.feature) + ')'

    def evaluate(self, data):
        return 1 / 1 + np.exp(-data)

class Sine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'sin(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.sin(data)

class Cosine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'cos(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.cos(data)

class Exp(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'exp(-abs(' + str(self.feature) + '))'

    def evaluate(self, data):
        return np.exp(-np.abs(data))

class Ln(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'ln(abs(' + str(self.feature) + ') + 1)'

    def evaluate(self, data):
        return np.log(np.abs(data) + 1)

class x72(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^7/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 7 / 2)

class x52(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^5/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 5 / 2)

class x13(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^1/3'

    def evaluate(self, data):
        return np.power(np.abs(data), 1 / 3)
