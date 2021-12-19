import numpy as np
from DeepNeuralNetworkARC import Activators

"""


FORWARD PASS:
    hidden_state0 = dot(Wxh, x) + b
    input_layer = activate(hidden_state0)
    hidden_layer = activate(dot(Whh, input_layer1) + b + hidden_state0)
    output _layer = dot(Why, hidden_layer) + bias
    
BACK PROPAGATION:
    #partial derivatives of the cost with respect to each weight
    dCost_dWhy = dot(output, h_transpose)v
    dCost_dWhh = (1 - h^2)h[t-1]
    dCost_dWxh = (1 - h^2)inp

"""


training_data = [[[i+j] for i in range(5)] for j in range(100)]
labels = [[i+5] for i in range(100)]

wxh = np.random.rand(5, 5)
whh = np.random.rand(5, 5)
why = np.random.rand(5, 5)
b = np.random.rand(5)
bo = np.random.rand(1)
hidden_state = [np.random.rand(5, 1)]

weights = [wxh, whh, why]

def foward_pass():
    for training_matrix in training_data:
        print(np.array(training_matrix).shape)
        hidden_state0 = np.array(np.dot(wxh, training_matrix))
        print(np.array(hidden_state[-1]).shape)
        hidden_state.append(Activators().sigmoid(hidden_state0 + np.dot(whh, hidden_state[-1]) + b))
        # print(np.array(hidden_state[-1]).shape)
        output_layer = np.dot(why, np.array(hidden_state[-1])) + bo
        print(output_layer.shape)


def backpropagation():
    for weight in weights:


if __name__ == '__main__':
    foward_pass()



#
# from matplotlib import pyplot
#
# x, b = np.array([_ for _ in range(10, 120)]), np.array([_ for _ in range(110)])
# y = 5
#
# x1_plot = (3*(x**2)*y) + (b**2)
# x2_plot = (6*x*y) + (2*b)
# pyplot.plot(x1_plot, c='r')
# pyplot.plot(x2_plot, c='b')
#
# pyplot.show()
#

