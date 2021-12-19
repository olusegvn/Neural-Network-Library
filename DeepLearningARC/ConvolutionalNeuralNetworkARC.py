import numpy as np
import pandas as pd
from DeepNeuralNetworkARC import *
from sklearn.preprocessing import normalize


class ConvolutionalNeuralNetwork:
    def pad(self, image, p=1):
        if len(image.shape) == 3:
            padded_frame = []
            for frame in image:
                padded_frame.append(self.pad(frame, p))
            return padded_frame
        height, width = image.shape
        padded_image = []
        [padded_image.append(np.zeros(image.shape[1])) for _ in range(p)]
        padded_image.extend(image)
        [padded_image.append(np.zeros(image.shape[1])) for _ in range(p)]
        Image = pd.DataFrame(padded_image)
        for _ in range(p):
            Image.insert(0, 0, np.zeros_like(height), True)
            Image.insert(width + p, 0, np.zeros_like(height), True)
        return Image.as_matrix()

    def convolve(self, image, _filter, stride=1, p=1):
        image = self.pad(image, p)
        if len(image.shape) == 3:
            convolved_image = []
            for filter_counter in range(image.shape[2]):
                convolved_image.append(self._convolve(image[filter_counter], _filter[filter_counter], stride, p=0))
            return convolved_image

    def pool(self, image, f, p, stride):# f: filter, p: padding
        image = self.pad(image, p)
        convolved_frame = []
        height, width = image.shape
        image = np.array(image)
        h_filter, w_filter = f.shape
        for row in range(((height + (2*p) - h_filter)/stride) + 1):
            for column in range(((width + (2*p) - w_filter)/stride) + 1):
                conv_reigion = image[row:(row + h_filter), column:(column + (column + w_filter))]
                convolved_frame[row][column] = np.max(conv_reigion)
                column += stride - 1
            row += stride - 1

    def function_by_reigion(self, image, _filter, stride=1, p=1): # p: padding
        image = self.pad(image, p)
        convolved_frame = []
        height, width = image.shape
        image, _filter = np.array(image), np.array(_filter)
        h_filter, w_filter = _filter.shape
        for row in range(((height + (2*p) - h_filter)/stride) + 1):
            for column in range(((width + (2*p) - w_filter)/stride) + 1):
                conv_reigion = image[row:(row + h_filter), column:(column + (column + w_filter))]
                convolved_frame[row][column] = sum(conv_reigion * _filter)
                column += stride - 1
            row += stride - 1
        return convolved_frame




if __name__ == '__main__':
    image = np.ones((3, 3, 3))
    NN = ConvolutionalNeuralNetwork()
    print(NN.pad(image, 0))
