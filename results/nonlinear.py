import numpy as np

class Non_linear():
    def __init__(self, feature):
        self.feature = feature

class Gauss(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'gauss(' + str(self.feature) + ')'
    def evaluate(self, data):
        return np.exp(-(data**2))
    
class Tanh(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'tanh(' + str(self.feature) + ')'
    def evaluate(self, data):
        return np.tanh(data)
    
class Atan(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'atan(' + str(self.feature) + ')'
    def evaluate(self, data):
        return np.arctan(data)
    
class Sigmoid(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'sigma(' + str(self.feature) + ')'

    def evaluate(self, data):
        return 1 / 1 + np.exp(-data)

class Sine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'sin(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.sin(data)

class Cosine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'cos(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.cos(data)

class Exp(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'exp(abs(' + str(self.feature) + '))'

    def evaluate(self, data):
        return np.exp(np.abs(data)+0.000001)

class Expm1(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'exp(abs(' + str(self.feature) + '))'

    def evaluate(self, data):
        return np.exp(-np.abs(data)+0.000001)

class Ln(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'ln(abs(' + str(self.feature) + '))'

    def evaluate(self, data):
        return np.log(np.abs(data)+0.000001)

class Ln1p(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'ln(abs(' + str(self.feature) + ') + 1)'

    def evaluate(self, data):
        return np.log1p(np.abs(data)+0.000001)
    
class x72(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^7/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 7 / 2)

class x52(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^5/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 5 / 2)

class xm15(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^-1/5'

    def evaluate(self, data):
        return np.power(np.abs(data)+0.00001, -1/5)

class x13(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^1/3'

    def evaluate(self, data):
        return np.power(np.abs(data), 1 / 3)

class PolyAbs(Non_linear):
    def __init__(self, feature, power):
        super().__init__(feature)
        self.power = power
    def __str__(self):
        return '(' + str(self.feature) + ')^' + str(self.power)

    def evaluate(self, data):
        return np.power(np.abs(data), self.power)

class Poly(Non_linear):
    def __init__(self, feature, power):
        super().__init__(feature)
        self.power = power
    def __str__(self):
        return '(' + str(self.feature) + ')^' + str(self.power)

    def evaluate(self, data):
        return np.power(data, self.power)
    

class p1(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001)

class pm1(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^-1'
    def evaluate(self, x):
        return (x+0.00001) ** (-1)
                      
class pm2(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^-2'
    def evaluate(self, x):
        return (x + 0.00001) **(-2)

class pm05(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(abs(' + str(self.feature) + '))^-1/2'
    def evaluate(self, x):
        return np.abs(x + 0.00001) ** (-0.5)


class p05(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(abs(' + str(self.feature) + '))^1/2'
    def evaluate(self, x):
        return np.abs(x) ** 0.5

class p2(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^2'
    def evaluate(self, x):
        return x ** 2

class p3(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return '(' + str(self.feature) + ')^3'
    def evaluate(self, x):
        return x ** 3

class p0p0(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))^2'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * np.log(np.abs(x) + 0.00001)
    
class p0pm1(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^-1'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * ((x + 0.00001) ** (-1))

class p0pm2(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^-2'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * ((x + 0.00001)** (-2))

class p0pm05(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^-1/2'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * (np.abs(x+0.00001) ** (-0.5))

class p0p05(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^1/2'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * (np.abs(x) ** (0.5))

class p0p1(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))' + str(self.feature)
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * x

class p0p2(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^2'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * (x ** 2)

class p0p3(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 2
    def __str__(self):
        return 'log(abs(' + str(self.feature) + '))(' + str(self.feature) + ')^3'
    def evaluate(self, x):
        return np.log(np.abs(x) + 0.00001) * (x ** 3)

class Imean(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
        self.comp = 1
    def __str__(self):
        return 'I(' + str(self.feature) + ' > mean(' + str(self.feature) + '))'
    def evaluate(self, x):
        return np.where(x >= np.mean(x), 1, 0)