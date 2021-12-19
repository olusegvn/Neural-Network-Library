import numpy as np


class Correlation:
    def __init__(self, x,  y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = self.x. shape[0]

    def correlation(self):
        n = self.n
        numerator = (n * (self.x * self.y).sum()) - (self.x.sum() * self.y.sum())
        denominator = lambda x: np.sqrt((n * np.square(x).sum()) - np.square(x.sum()))
        return numerator/(denominator(self.x) * denominator(self.y))

    def regression_fit(self):
        n = self.n
        numerator = (n * ((self.x * self.y).sum())) - (self.x.sum() * self.y.sum())
        denominator = (n * np.square(self.x).sum()) - np.square(self.x.sum())
        b = numerator/denominator
        a = self.y.mean() - (b * self.x.mean())
        return a, b


x = np.array([[4.2, 1.4, 6.6, 4.7, 2.6, 5.8, 1.8, 5.8, 7.3, 6.4]]).reshape(10, 1)
y = np.array([[1.9, 0.7, 2.2, 2.0, 1.1, 2.6, 0.3, 2.3, 2.6, 2.4]]).reshape(10, 1)

print(Correlation(x, y).regression_fit())

