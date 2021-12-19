import math
import pickle
import os
import time
import numpy as np
from datetime import timedelta
from joblib import dump, load
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

'''TEDRARC Neural Network architecture containing:
        [*]Activation functions   :- sigmoid
                                   - other
                                   - Binary step   
                                   - TANH/ Hyperbolic tangent
        [*]Activation derivatives :- sigmoid derivative 
                                   - TANH derivative
                                   - linear derivative
        [*]loss functions         :- simple
                                   - sum of squares  
        [*]catching and delivery of best weights
'''


class Activators:
    def linear(self, x, constant):
        """formula : cx
        - unusable in backpropagation
        - derivative = constant
        """
        activator = constant * x
        activator_der = constant
        return activator, activator_der

    def binary_step(self, input):
        return np.round(self.sigmoid(input)[0]), np.round(self.sigmoid(input)[1])

    def sigmoid(self, x, *argv):
        """
        non-linear activation on scale 0-1
        """
        sigmoid = 1 / (1 + np.e ** -x)
        sigmoid_der = sigmoid * (1 - sigmoid)
        return sigmoid, sigmoid_der

    def TANH(self, x, *TANH_args):
        """
        zero-centered activation function on scale -1 to 1
        :param x:
        :return: tanh x
        """
        sinhx = ((np.e ** x) - (np.e ** -x)) / 2
        coshx = ((np.e ** x) + (np.e ** -x)) / 2
        tanh = sinhx / coshx
        tanh_der = 1 - pow(tanh, 2)
        return tanh, tanh_der

    def ReLU_der(self, x, ReLU_args):
        x[x < 0] = ReLU_args['l_threshold']
        x[x > 0] = ReLU_args['u_threshold']
        try:
            x[x == 0] = ReLU_args[2]
        except:
            x[x == 0] = 0.5

        return x

    def ReLU(self, x, ReLU_args):

        """
        Rectified Linear Unit
        scale: 0 - infinity
        :param x: array of elements to be activated.
        :param threshold: value which elements below would be conerted to.
        :return: max(x, 0), ReLU derivative
        implementation of parametric ReLU comming soon
        """
        return np.maximum(ReLU_args['l_threshold'], x), self.ReLU_der(x, ReLU_args)

    def softmax(self, x, *args):
        x = np.array(x)
        softmax =  np.exp(x)/sum(np.exp(x))
        softmax_der = softmax * (1 - softmax)
        return softmax, softmax_der


class Cost:
    def difference(self, z, label):
        error = z - label
        error_der = z - label
        return error, error_der

    def sum_of_squares(self, z, label):
        error = pow((z - label), 2)
        error_der = 2 * (z - label)
        return error, error_der

    def cross_entropy(self, z, label):
        entropy = -(1 / len(z)) * np.log(z.sum())
        entropy_der = np.log
        return entropy, entropy_der

    def absolute_error_loss(self, z, label):
        error = np.abs(z - label)
        error_der = np.abs(z - label)
        return error, error_der

    def mean_absolute_error(self, z, label):
        error = np.mean(self.absolute_error_loss(z, label)[0])
        return error

    def mean_squared_error(self, z, label):
        error = np.mean(self.sum_of_squares(z, label)[0])
        return error


def feedforward(model, input, activator, activator_args):
    for layer in model:
        weights, bias = model[layer]['weights'], model[layer]['bias']
        z = activator((np.dot(input, weights) + bias), activator_args)[0]
        input = z
    return z


class Optimizers:
    def gradient_descent(self, model, input, labels, cost, lr, ADAGRAD, e, activator, activator_args):
        gradients = dict()
        # Backpropagation for each layer in the model
        z = feedforward(model, input, activator, activator_args)
        error = cost(z, labels)
        for layer in model:
            weights, bias = model[layer]['weights'], model[layer]['bias']
            gradients[layer] = []
            gradients[layer + 'bias'] = []
            dcost_dz, dz_dy, dy_dw = error[1], activator(z, activator_args)[1], input.T
            d1 = dz_dy * dcost_dz
            dcost_dw = np.dot(dy_dw, d1)

            if ADAGRAD:
                try:
                    gradient = gradients[layer][-1]
                except IndexError:
                    gradients[layer].append(np.zeros_like(dcost_dw))
                    gradients[layer + 'bias'].append(np.zeros_like(d1))
                bias_gradient = gradients[layer + 'bias'][-1]
                gradient = gradients[layer][-1]
                gradients[layer + 'bias'].append(bias_gradient + d1 ** 2)
                gradients[layer].append(gradient + (dcost_dw ** 2))
                weights -= (lr / np.sqrt(gradient + e)) * dcost_dw
                _d1 = (lr / np.sqrt(bias_gradient + e)) * d1
                for value in _d1:
                    bias -= value
            if not ADAGRAD:
                weights -= dcost_dw * lr
                for value in d1:
                    bias -= value * lr
            input = z
        return error[0], model


class TRNeuralNetwork():
    def __init__(self, model={}, type='feedforward', fine=False):
        """other types of neural networks : - perception
                                            - feedforwawd
                                            - Convolutional
                                            - Deep
        """
        self.type = type
        self.model = model
        self.fine = fine
        self.lr = 0
        self.error_history = []
        self.best_model = []
        self.activator_args = {}
        self.activator = lambda v: []
        self.high_lr = False

    def play_randomize(self, x, y, no_of_trials, activator=Activators.sigmoid, cost=Cost().sum_of_squares):
        best_error = 100
        for _ in range(no_of_trials):
            for layer in self.model:
                weights, bias = model[layer]['weights'], model[layer]['bias']
                weights, bias = np.random.rand(len(weights)), np.random.rand(len(bias))
                z = activator(np.dot(x, weights) + bias)[0]
                error = cost(z, y)[0]
            if error.sum() < best_error:
                best_error = error.sum()
                self.best_model = self.model
            print('Error : ', error.sum(), 'Best error : ', best_error)
            self.cache('randomized_model.pickle')

    def train(self, x, y, epochs=10000, optimizer=Optimizers().gradient_descent, ADAGRAD=True, e=10 ** -6,
              activator=Activators().sigmoid, show_step=1, cost=Cost().difference, learning_rate=0.1,
              cut_pickle='model_pickle.pickle', batch_rows=1, activator_args={}, show_error_history=False, **kwargs):
        self.lr = learning_rate
        self.activator, self.activator_args = activator, activator_args
        batch_rows = batch_rows if batch_rows else len(x)
        if self.type in ['simple', 'perceptron']:
            self.model = {'layer1': {'weights': np.random.rand(self.model, 1),
                                     'bias': np.random.rand(1)}}
        best_error = 100

        if self.fine:
            for layer in self.model:
                weights, bias = self.model[layer]['weights'], self.model[layer]['bias']
                weights, bias = (2 * weights) - 1, bias
        try:
            for _ in range(epochs):
                start_time = time.time()
                total_error, batch_start, batch_end = [], 0, batch_rows
                for __ in range(int(len(x) / batch_rows)):
                    _error, self.model = optimizer(self.model, x[batch_start: batch_end], y[batch_start: batch_end],
                                                   cost, learning_rate, ADAGRAD, e, activator=activator,
                                                   activator_args=activator_args, )
                    # e: numerical value added to ADAGRAD gradient layer
                    # to maintain numerical stability
                    total_error.append(_error.sum())
                    batch_start += batch_rows
                    batch_end += batch_rows
                error = np.mean(np.abs(total_error))
                self.error_history.append(error)
                if error == min(self.error_history):
                    best_error = error
                    self.best_model = self.model
                if abs(error) > abs(best_error):
                    self.high_lr = True
                if (_ + 1) % show_step == 0 or (_ + 1) == 1:
                    print('Epoch: ', _ + 1, '\t|  Error: ', error, end='\t')
                    seconds = ((time.time() - start_time) * (epochs - _))
                    print('ETA : ', timedelta(seconds=seconds))
                    plt.plot(self.error_history)
                    plt.show()
                    self.cache(cut_pickle)
        except KeyboardInterrupt:
            self.cache(cut_pickle)

        if self.high_lr:
            print('\n\033[91m[*] Learning rate too high, reduce for better performance\033[0m')
            print('\n\033[94mbest_error : ', best_error, '\033[0m')

    def pick(self, pickle_file):
        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as f:
                # try:
                #     self.best_model, self.activator, self.activator_args, self.lr, self.error_history = pickle.load(f)
                # except:
                self.best_model = pickle.load(f)
                self.model = self.best_model

    def predict(self, input, nrows=-1):
        # if nrows == -1:
        #     batch_start, batch_end, result = 0, batch_rows, []
        # for __ in range(int(len(input) / nrows)):
        #     result.append(feedforward(self.best_model, input[batch_start:batch_end], self.activator, self.activator_args))
        #     batch_start += nrows
        #     batch_end += nrows
        # return np.array(result).reshape(len(input), 1)
        return feedforward(self.best_model, input, self.activator, self.activator_args)
    
    def cache(self, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump([self.best_model, self.activator, self.activator_args, self.lr, self.error_history], f)


# x = np.array([[0.2, 0.1, 0.0], [0.3, 0.0, 0.1], [0.1, 0.2, 0.2], [0.3, 0.1, 0.0], [0.2, 0.1, 0.1]])
# y = np.array([[0.1, 0.2, 0.0, 0.1, 0.2]])
# y = y.reshape(5, 1)
# if __name__ == '__main__':
#     model = {
#         'layer1': {'weights': np.random.rand(3, 1),
#                    'bias': np.random.rand(1)},
#     }
#     NN = TRNeuralNetwork(model, type='feedforward', fine=True)
#     NN.train(x, y, activator=Activators().ReLU, cost=Cost().sum_of_squares, epochs=10000, learning_rate=0.0001,
#                  activator_args={'u_threshold': 1, 'l_threshold': 0}, ADAGRAD=True, batch_rows=1, show_step=1000)
#     print(np.round(NN.predict([[0.2, 0.1, 0.0], [0.3, 0.0, 0.1], [0.1, 0.2, 0.2], [0.3, 0.1, 0.0], [0.2, 0.1, 0.1]]), 2))
#     plt.plot(NN.error_history)
#     plt.ylim(0.0, 0.6)
#     plt.show()
# if __name__ == '__main__':
#     nodes = 3
#     NN = TRNeuralNetwork(nodes, type='perception', fine=True)
#     NN.train(x, y, activator=Activators().sigmoid, cost=Cost().sum_of_squares, epochs=17050, learning_rate=0.1)
#     NN.predict([0.2, 0.1, 0.1])
meta_model_x_train, meta_model_y_train, meta_model_x_test, meta_model_y_test = load('meta.joblib')
model = {
        'layer3': {'weights': np.random.rand(10, 5),
                   'bias': np.random.rand(1)},
        'layer2': {'weights': np.random.rand(5, 3),
                   'bias': np.random.rand(1)},
        'layer1': {'weights': np.random.rand(3, 1),
                   'bias': np.random.rand(1)},
    }
meta_model = TRNeuralNetwork(model, type='feedforward', fine=True)
meta_model.train(meta_model_x_train, meta_model_y_train, activator=Activators().ReLU, cost=Cost().sum_of_squares, epochs=100000, learning_rate=0.000001,
             activator_args={'u_threshold': 1, 'l_threshold': 0}, ADAGRAD=False, batch_rows=None, show_step=1000)

print(balanced_accuracy_score(meta_model_y_train, np.round(meta_model.predict(meta_model_x_train))))
m = tf.keras.metrics.Recall()
m.update_state(meta_model_y_train, np.round(meta_model.predict(meta_model_x_train)))
print(m.result().numpy())

m = tf.keras.metrics.Precision()
m.update_state(meta_model_y_train, np.round(meta_model.predict(meta_model_x_train)))
print(m.result().numpy())

print(balanced_accuracy_score(meta_model_y_test, np.round(meta_model.predict(meta_model_x_test))))
m = tf.keras.metrics.Precision()
m.update_state(meta_model_y_test, np.round(meta_model.predict(meta_model_x_test)))
print(m.result().numpy())
m = tf.keras.metrics.Recall()
m.update_state(meta_model_y_test, np.round(meta_model.predict(meta_model_x_test)))
print(m.result().numpy())
