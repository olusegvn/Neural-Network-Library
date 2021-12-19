import datetime
import time
from statistics import mode
from collections import Counter
import numpy as np

def square(_list):
    return [pow(_, 2) for _ in _list]


class KNeighborsClassifier:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.results = []

    def predict(self, array, k=3, show_step=1):
        counter = 1
        for _ in range(len(array)):
            distances = []
            start_time = time.time()
            for trainer in range(len(self.data)):
                distance = np.linalg.norm(array[_] - self.data[trainer])
                distances.append([distance, self.labels[trainer]])
            # print('there in ', time.time() - f)
            neighbors = [i[1] for i in sorted(distances)[:k]]
            neighbor = Counter(neighbors).most_common(1)[0][0]
            self.results.append(neighbor)
            _time = time.time() - start_time
            if counter % show_step == 0:
                print(round((counter/len(array)) * 100, 2), '%  :\t', counter, 'of', len(array),
                      'classified |\t ETA : ', datetime.timedelta(seconds=_time * (len(array) - counter)))
            counter += 1
        return self.results

    def check_accuracy(self, labels):
        error = (np.array(self.results) - np.array(labels)).sum()
        accuracy = 100 - (error * 100)
        return accuracy, '%'



